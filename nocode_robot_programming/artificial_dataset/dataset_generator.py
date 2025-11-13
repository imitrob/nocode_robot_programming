import os, json, math, argparse
import numpy as np
import trimesh, pyrender, imageio

from dataclasses import dataclass

import nocode_robot_programming
import trajectory_data # Package where the dataset is saved
ARTIFICIAL_DATASET_PATH = f"{trajectory_data.package_path}/trajectories/artificial_dataset"

def parse_offset(s):
    try:
        dx, dy, dz = [float(x.strip()) for x in s.split(",")]
        return np.array([dx, dy, dz], dtype=np.float32)
    except Exception:
        raise argparse.ArgumentTypeError("Offset must be 'dx,dy,dz'")

def look_at(eye, target=np.array([0,0,0.0]), up=np.array([0,0,1.0])):
    """
    Build a camera-to-world pose matrix for pyrender.
    - eye: 3-vector camera position in world
    - target: 3-vector point the camera looks at
    - up: global up direction
    """
    f = (target - eye)
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(u := up)
    s = np.cross(f, u); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    T = np.eye(4, dtype=np.float32)
    T[0, :3] = s
    T[1, :3] = u
    T[2, :3] = -f
    T[:3, 3] = eye
    return T

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_trajectory_json(path):
    """ Accepts JSON list with ANY of:
        - {"pose_cam_to_world": [[...4],...4], "target":[x,y,z]?}
        - {"eye":[x,y,z], "target":[x,y,z]?}
        - raw 4x4 arrays (list of 16 or 4x4)
    Returns: poses(list of 4x4), eyes(list of 3), targets(list of 3)
    """
    data = json.load(open(path, "r"))
    poses, eyes, targets = [], [], []
    for item in data:
        if isinstance(item, dict) and "pose_cam_to_world" in item:
            P = np.array(item["pose_cam_to_world"], dtype=np.float32).reshape(4,4)
            poses.append(P)
            eyes.append(P[:3, 3].astype(np.float32))
            tgt = np.array(item.get("target", [0.,0.,0.]), dtype=np.float32)
            targets.append(tgt)
        elif isinstance(item, dict) and "eye" in item:
            eye = np.array(item["eye"], dtype=np.float32)
            tgt = np.array(item.get("target", [0.,0.,0.]), dtype=np.float32)
            P = look_at(eye, tgt)
            poses.append(P); eyes.append(eye); targets.append(tgt)
        else:
            P = np.array(item, dtype=np.float32).reshape(4,4)
            poses.append(P)
            eyes.append(P[:3,3])
            targets.append(np.array([0.,0.,0.], dtype=np.float32))
    return poses, eyes, targets

def build_default_approach(n_frames, start_dist, end_dist, height):
    # Move along +Y towards origin at fixed Z=height, always looking at origin.
    d = np.linspace(start_dist, end_dist, n_frames).astype(np.float32)
    eyes = np.stack([np.zeros_like(d), d, np.full_like(d, height)], axis=1)  # (N,3)
    target = np.array([0., 0., 0.], dtype=np.float32)
    poses = [look_at(eye, target) for eye in eyes]
    targets = [target.copy() for _ in range(n_frames)]
    return poses, eyes, targets

def generate_run(args):

    # Load mesh
    absolute_path_mesh = f"{nocode_robot_programming.package_path}/nocode_robot_programming/artificial_dataset/{args.mesh}"
    mesh = trimesh.load(absolute_path_mesh, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) and isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Failed to load a mesh geometry.")

    # Normalize/center at origin
    mesh.apply_translation(-mesh.bounding_box.centroid)

    max_extent = float(np.max(mesh.extents))
    if max_extent <= 0:
        max_extent = 1.0

    # Camera height derived from model size
    H = args.height_factor * max_extent

    # Renderer & scene
    bg = np.array([float(x) for x in args.bg_rgba.split(",")], dtype=np.float32)
    if bg.size != 4:
        raise ValueError("--bg_rgba must have 4 components 'r,g,b,a' in 0..1")

    scene = pyrender.Scene(bg_color=bg, ambient_light=[0.05, 0.05, 0.05])
    pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(pm)

    # Lighting
    def _mk_pose(p):
        return look_at(np.array(p, dtype=np.float32), np.zeros(3))
    key = pyrender.DirectionalLight(intensity=3.0)
    fill = pyrender.DirectionalLight(intensity=1.6)
    rim = pyrender.DirectionalLight(intensity=0.8)
    scene.add(key, pose=_mk_pose([ 5,  5, 10]))
    scene.add(fill, pose=_mk_pose([-10, -5, 15]))
    scene.add(rim,  pose=_mk_pose([-5, 10, 10]))

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(args.fov))
    renderer = pyrender.OffscreenRenderer(viewport_width=args.res, viewport_height=args.res)
    cam_node = scene.add(camera, pose=np.eye(4))  # updated per frame

    # Output dirs
    absolute_path = f"{ARTIFICIAL_DATASET_PATH}/{args.folder}"
    root = ensure_dir(absolute_path)
    out_dir = ensure_dir(os.path.join(root, args.cls))

    if getattr(args, "traj_file", None):
        poses_all, eyes_all, targets_all = load_trajectory_json(args.traj_file)
        if len(poses_all) == 0:
            raise RuntimeError("Empty trajectory file.")
        N = min(args.n_frames, len(poses_all)) if args.n_frames > 0 else len(poses_all)
        idxs = np.linspace(0, len(poses_all) - 1, N).astype(int)
        poses = [poses_all[i] for i in idxs]
        eyes  = [eyes_all[i]  for i in idxs]
        targets = [targets_all[i] for i in idxs]
    else:
        start_dist = args.start_dist_factor * max_extent
        end_dist   = args.end_dist_factor   * max_extent
        poses, eyes, targets = build_default_approach(args.n_frames, start_dist, end_dist, H)

    offset_vec = np.array(args.offset, dtype=np.float32)

    gray_frames = []
    meta_dict = {}
    for i, (P_base, eye_base, tgt) in enumerate(zip(poses, eyes, targets)):
        # Apply translation offset to camera position; keep the look aimed at target
        eye = eye_base + offset_vec
        P = look_at(eye, tgt)

        scene.set_pose(cam_node, pose=P)

        color, depth = renderer.render(scene)
        alpha = (depth > 0).astype(np.uint8) * 255
        rgba = np.dstack([color, alpha])

        # grayscale (uint8), background -> 0 using alpha
        rgb_f = rgba[..., :3].astype(np.float32)
        gray = (0.2126 * rgb_f[..., 0] + 0.7152 * rgb_f[..., 1] + 0.0722 * rgb_f[..., 2])
        gray[alpha == 0] = 0
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        gray_frames.append(gray_u8)  # (H, W)

        fn = f"view_{i:03d}.png"
        imageio.imwrite(os.path.join(out_dir, fn), rgba)

        meta_dict[fn] = {
            "eye": eye.tolist(),
            "target": tgt.tolist(),
            "pose_cam_to_world": P.tolist(),
            "fov_deg": float(args.fov),
            "width": int(args.res),
            "height": int(args.res),
            "base_eye": eye_base.tolist(),
            "base_pose_cam_to_world": P_base.tolist(),
        }
    g = np.stack(gray_frames, axis=0)
    with open(os.path.join(out_dir,  "metadata.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)
    np.savez_compressed(os.path.join(out_dir,  "grayscale_uint8.npz"), images=g)

    renderer.delete()

@dataclass
class RunArgs:
    mesh: str = "taskboard.stl"
    folder: str = "train"
    cls: str = "no_offset"
    n_frames: int = 20
    res: int = 224
    fov: float = 45.0
    height_factor: float = 1.1
    offset: tuple = (0,0.0,0.0)
    bg_rgba: str = "0,0,0,0"
    traj_file: str | None = None
    start_dist_factor: float = 1.5
    end_dist_factor: float = 0.2

