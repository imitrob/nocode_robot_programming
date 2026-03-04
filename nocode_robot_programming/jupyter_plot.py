from IPython.display import display, HTML
import matplotlib.pyplot as plt
import io
import base64

class JupyterPlot():
    """ Plot in Jupyter notebook. Plots aligned next to each other
    Usage:
        plt.plot(plot_data)
        self.save() # this saves the fig so another can be created    
        plt.plot(second_plot_data)
        self.save()
        plt.plot(third_plot_data)
        self.save()    
        self.show() # flushes the saved figures on a screen and visualize them
    """
    def __init__(self):
        self.plots = []
    def save(self):
        self.plt_save()
    def plt_save(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        if len(img_base64) > 3196:
            self.plots.append(img_base64)

    def delete(self):
        self.plots = []
    
    def show(self, small=False):
        if small:
            min_width = 15
        else:
            min_width = 35
        html = ""
        n = len(self.plots)
        if n == 0: return
        for plot in self.plots:
            html += f'<img src="data:image/png;base64,{plot}" style="width:{min(min_width,(100 // n) - 1)}%;display:inline-block;margin:5px;">'
        display(HTML(html))
        self.plots = []

jupyter_plot = JupyterPlot()


# Inline playback in Jupyter using Matplotlib (no ffmpeg/GIF)
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display

def show_gray_video_cuda(frames_cuda: torch.Tensor, fps: int = 20, repeat: bool = True, scale: float = 1.0):
    """
    Play a grayscale video tensor [T, H, W] (float32, 0..1) that lives on CUDA.
    Renders inline using JS (no external encoders).
    """
    assert frames_cuda.ndim == 3, "Expected [T, H, W] grayscale tensor"
    # Move to CPU and convert to uint8 for fast imshow
    frames_u8 = (frames_cuda.detach().clamp(0,1) * 255).round() \
                .to(torch.uint8).cpu().numpy()  # shape: [T, H, W]

    T, H, W = frames_u8.shape
    # fig, ax = plt.subplots()
    dpi = 100  # any dpi; we size the figure to match pixels
    fig, ax = plt.subplots(
        figsize=(scale * W / dpi, scale * H / dpi),  # ← scale applied here
        # figsize=(W / dpi, H / dpi), dpi=dpi, frameon=False,  # no figure frame
        constrained_layout=False
    )
    fig.patch.set_alpha(0)  # transparent figure background
    ax.set_position([0, 0, 1, 1])  # axes fills entire figure (no margins)
    ax.set_axis_off()
    im = ax.imshow(frames_u8[0], cmap='gray', animated=True, vmin=0, vmax=255)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    def _update(i):
        im.set_array(frames_u8[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=1000/fps, blit=True, repeat=repeat
    )
    plt.close(fig)  # avoid double display
    return HTML(f'<div style="margin:0;padding:0;line-height:0">{anim.to_jshtml()}</div>')


import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from typing import Sequence, Optional, Callable, Union

def show_gray_video_cuda_captions(
    frames_cuda: torch.Tensor,
    fps: int = 20,
    repeat: bool = True,
    scale: float = 1.0,
    captions: Optional[Union[Sequence[str], Callable[[int], str]]] = None,
    caption_xy: tuple = (0.02, 0.95),     # (x,y) in axes coords (0..1)
    caption_fontsize: Optional[int] = None,
    caption_color: str = "white",
    caption_bbox: Optional[dict] = None,  # e.g. {"facecolor":"black","alpha":0.6,"pad":0.4,"boxstyle":"round"}
):
    """
    Play a grayscale video tensor [T, H, W] (float32, 0..1) that lives on CUDA.
    Renders inline using JS (no external encoders). Supports per-frame captions.

    captions:
      - sequence of length T with strings, OR
      - callable: captions(i) -> str
    """
    assert frames_cuda.ndim == 3, "Expected [T, H, W] grayscale tensor"

    # Move to CPU and convert to uint8 for fast imshow
    frames_u8 = (frames_cuda.detach().clamp(0, 1) * 255).round() \
                .to(torch.uint8).cpu().numpy()  # [T, H, W]

    T, H, W = frames_u8.shape

    # Resolve captions list (or None)
    captions_list: Optional[Sequence[str]] = None
    if captions is not None:
        if isinstance(captions, torch.Tensor):
            captions_list = [str(captions[i].item()) for i in range(T)]
        elif callable(captions):
            captions_list = [str(captions(i)) for i in range(T)]
        else:
            captions_list = list(captions)
            assert len(captions_list) == T, "captions must have length T"

    # Figure/axes sized to pixel dimensions (scaled)
    dpi = 100
    fig, ax = plt.subplots(
        figsize=(scale * W / dpi, scale * H / dpi),
        constrained_layout=False
    )
    fig.patch.set_alpha(0)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()

    # Image artist
    im = ax.imshow(frames_u8[0], cmap='gray', animated=True, vmin=0, vmax=255)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    # Caption artist (optional)
    txt = None
    if captions_list is not None:
        if caption_bbox is None:
            caption_bbox = {"facecolor": "black", "alpha": 0.6, "pad": 0.4, "boxstyle": "round"}
        if caption_fontsize is None:
            # heuristic based on height
            caption_fontsize = max(10, int(0.045 * H * scale))

        txt = ax.text(
            caption_xy[0], caption_xy[1],
            captions_list[0],
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=caption_fontsize,
            color=caption_color,
            bbox=caption_bbox,
            animated=True  # important for blitting
        )

    def _update(i):
        im.set_array(frames_u8[i])
        if txt is not None:
            txt.set_text(captions_list[i])
        return (im, txt) if txt is not None else (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=1000.0 / fps, blit=True, repeat=repeat
    )
    plt.close(fig)  # avoid double display
    return HTML(f'<div style="margin:0;padding:0;line-height:0">{anim.to_jshtml()}</div>')

def show_gray_video_cuda_captions_aligned(
    frames_cuda: torch.Tensor,
    fps: int = 20,
    repeat: bool = True,
    scale: float = 1.0,
    captions: Optional[Union[Sequence[str], Callable[[int], str]]] = None,
    caption_xy: tuple = (0.02, 0.95),     # (x,y) in axes coords (0..1)
    caption_fontsize: Optional[int] = None,
    caption_color: str = "white",
    caption_bbox: Optional[dict] = None,  # e.g. {"facecolor":"black","alpha":0.6,"pad":0.4,"boxstyle":"round"}
    Xt: Optional[Union[Sequence[int], torch.Tensor, np.ndarray]] = None,
    max_rows: int = 10,
):
    """
    Play a grayscale video tensor [T, H, W] (float32, 0..1) that lives on CUDA.
    If Xt is provided (shape [T], int timesteps), the animation runs over
    *unique timesteps*, and for each timestep we show a vertical stack of up
    to `max_rows` images that share that timestep.

    - Xt: integer timestep per frame, len T
    - max_rows: max number of stacked images per timestep (rest blank)
    """
    assert frames_cuda.ndim == 3, "Expected [T, H, W] grayscale tensor"

    # Move to CPU and convert to uint8 for fast imshow
    frames_u8 = (frames_cuda.detach().clamp(0, 1) * 255).round() \
                .to(torch.uint8).cpu().numpy()  # [T, H, W]

    T, H, W = frames_u8.shape

    # Resolve captions list (or None)
    captions_list: Optional[Sequence[str]] = None
    if captions is not None:
        if isinstance(captions, torch.Tensor):
            captions_list = [str(captions[i].item()) for i in range(T)]
        elif callable(captions):
            captions_list = [str(captions(i)) for i in range(T)]
        else:
            captions_list = list(captions)
            assert len(captions_list) == T, "captions must have length T"

    if Xt is not None:
        # Convert Xt to numpy int array
        if isinstance(Xt, torch.Tensor):
            Xt_np = Xt.detach().cpu().numpy()
        else:
            Xt_np = np.asarray(Xt)
        Xt_np = Xt_np.astype(int)

        assert Xt_np.shape[0] == T, "Xt must have length T"

        unique_ts = np.unique(Xt_np)             # sorted timesteps
        num_steps = len(unique_ts)
        W_stack = W * max_rows                   # vertical stack height

        stacked_frames = np.zeros(
            (num_steps, H, W_stack), dtype=np.uint8
        )
        stacked_captions: Optional[list[str]] = [] if captions_list is not None else None

        for step_idx, t_val in enumerate(unique_ts):
            idxs = np.where(Xt_np == t_val)[0]
            n_here = min(len(idxs), max_rows)

            # Fill stacked frame rows with images for this timestep
            for row in range(n_here):
                src_idx = idxs[row]
                stacked_frames[step_idx,
                               :, row * W:(row + 1) * W] = frames_u8[src_idx]

            # Caption for this timestep-frame
            if captions_list is not None:
                caps = [captions_list[i] for i in idxs[:n_here]]
                caption_text = f"t={t_val} | " + " | ".join(caps)
                stacked_captions.append(caption_text)

        # Replace originals with stacked-per-timestep frames
        frames_u8 = stacked_frames
        T, H, W = frames_u8.shape

        if stacked_captions is not None:
            captions_list = stacked_captions
        else:
            # At least say the timestep and number of images
            captions_list = [f"t={t_val} ({np.sum(Xt_np == t_val)} frames)"
                             for t_val in unique_ts]

    dpi = 100
    fig, ax = plt.subplots(
        figsize=(scale * W / dpi, scale * H / dpi),
        constrained_layout=False
    )
    fig.patch.set_alpha(0)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()

    # Image artist
    im = ax.imshow(frames_u8[0], cmap='gray', animated=True, vmin=0, vmax=255)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    # Caption artist (optional)
    txt = None
    if captions_list is not None:
        if caption_bbox is None:
            caption_bbox = {"facecolor": "black", "alpha": 0.6, "pad": 0.4, "boxstyle": "round"}

        txt = ax.text(
            caption_xy[0], caption_xy[1],
            captions_list[0],
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=7,
            color=caption_color,
            bbox=caption_bbox,
            animated=True  # important for blitting
        )

    def _update(i):
        im.set_array(frames_u8[i])
        if txt is not None and captions_list is not None:
            txt.set_text(captions_list[i])
        return (im, txt) if txt is not None else (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=1000.0 / fps, blit=True, repeat=repeat
    )
    plt.close(fig)  # avoid double display
    return HTML(f'<div style="margin:0;padding:0;line-height:0">{anim.to_jshtml()}</div>')
