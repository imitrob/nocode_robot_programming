
import rclpy, time
from rclpy.node import Node
import argparse
import threading
import datetime
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from cv_bridge import CvBridgeError, CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from lfd_msgs.srv import StringService

from nocode_robot_programming.state_decision_dataset_prepare.dataset_auto import TrajectoryDatasetEvaluationViewBuilder
from skills_manager.ros_utils import SpinningRosNode
from nocode_robot_programming.state_decision.utils import visualize_video_frame_with_text, Filename
from nocode_robot_programming.state_decision.state_decider_model_manager import StateDeciderModelManager

WARNING_WHEN_IMAGE_OLDER_THAN = 0.2 # sec
MAX_TRAIN_TIME = 60.0 # sec

class StateDeciderNode(SpinningRosNode):
    def __init__(self, method: str):
        super(StateDeciderNode, self).__init__()

        if method == "SIFT":
            from nocode_robot_programming.state_decision.SIFT_model import StateDeciderSIFT
            model_factory = StateDeciderSIFT
        elif method == "DINO":
            from nocode_robot_programming.state_decision.dino_model import DINOFeaturePresence
            model_factory = DINOFeaturePresence
        elif method == "AEGP":
            from nocode_robot_programming.state_decision.AEGP_model import AEGP
            model_factory = AEGP
        elif method == "MANUAL":
            from nocode_robot_programming.state_decision.manual_state_decider import StateDeciderManual
            model_factory = StateDeciderManual
        else: raise Exception(f"Method '{method}' is not implemented!")

        self.method = method
        self.model_manager = StateDeciderModelManager(model_factory)

        self.create_service(StringService, "/state_decider_retrain", self.train_call, qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT), callback_group=self.callback_group)
        self.create_subscription(Image, "/modified_img", self.image_callback, 5)
        self.bridge = CvBridge()
        self.state_pub = self.create_publisher(String, "/target_state", 5)

        self.timestep = 0
        self.curr_image = None
        self.curr_image_timestamp = 0.0

        self.training_request = threading.Event()
        self.training_finished = threading.Event()

        self.task_name: str | None = None
        self.part_name: str | None = None
        self._done_parts: set[str] = set()
        self.new_run_started: bool = False

        self.dataset_builder = TrajectoryDatasetEvaluationViewBuilder()

    def train_call(self, msg, res):
        self.task_name = msg.text
        self.training_finished.clear()
        self.training_request.set()
        if not self.training_finished.wait(MAX_TRAIN_TIME):
            print("CHECK SEE THIS MESSAGE!!!!", flush=True)
            raise Exception("Training not finished in time, adjust MAX_TRAIN_TIME")
        print("TRAIN CALL FINISHED!!!", flush=True)
        return res

    def train(self):
        assert self.task_name is not None
        self._done_parts = set()

        self.dataset_builder = TrajectoryDatasetEvaluationViewBuilder()
        datasets, all_dataset = self.dataset_builder.load_deploy_from_task(self.task_name)

        self.model_manager.train(datasets, all_dataset)

    def image_callback(self, msg):
        try:
            resized_img_gray = self.bridge.imgmsg_to_cv2(msg)
            self.curr_image = resized_img_gray.reshape((1, resized_img_gray.shape[0], resized_img_gray.shape[1]))
            data = str(msg.header.frame_id).split("|")
            part_name = str(data[1])
            self.timestep = int(data[0]) + Filename(part_name).offset
            if part_name != self.part_name:
                if Filename(part_name).offset == 0:
                    # Root demo starting — new execution run, clear visited set
                    self._done_parts = set()
                    self.new_run_started = True
                else:
                    # Branching away from current part — mark it as done
                    if self.part_name is not None:
                        self._done_parts.add(self.part_name)
                self.part_name = part_name
            self.curr_image_timestamp = time.time()
        except CvBridgeError as e:
            print(e)

    def predict(self) -> tuple[str, str]:
        if self.curr_image is None:
            return "no img", "no image from /modified_img"

        note = ""
        if (time.time() - self.curr_image_timestamp) > WARNING_WHEN_IMAGE_OLDER_THAN:
            age = time.time() - self.curr_image_timestamp
            note = f"img stale {age:.1f}s"

        if self.method == "MANUAL":
            target_name = self.model_manager.manual_predict(self, self.timestep, self.task_name, self.part_name)
        else:
            target_name, model_note = self.model_manager.predict(self.curr_image, self.timestep)
            if model_note:
                note = f"{note}, {model_note}" if note else model_note

        if target_name == "nomodel":
            return target_name, note

        if target_name in self._done_parts:
            return "continue", f"skip (already visited {target_name})"

        self.state_pub.publish(String(data=target_name))
        return target_name, note

def main():
    parser = argparse.ArgumentParser(description="State Decider Node")
    parser.add_argument('--name_method', type=str, help='SIFT/DINO/AEGP/MANUAL', choices=["SIFT", "DINO", "AEGP", "MANUAL"], default="MANUAL")
    args = parser.parse_args()

    rclpy.init()
    node = StateDeciderNode(args.name_method)

    last_target = None
    last_stale = False  # tracks whether img was stale on the previous iteration

    while rclpy.ok(): # predict thread
        t0 = time.perf_counter()

        if node.training_request.is_set():
            node.training_request.clear()
            print("\n============ Training in progress ============", flush=True)
            node.train()
            node.training_finished.set()
            print("============ Training finished ============", flush=True)
            last_target = None
            last_stale = False

        if node.new_run_started:
            node.new_run_started = False
            print("\n============ New run started ============", flush=True)
            last_target = None
            last_stale = False

        target_name, note = node.predict()
        if node.curr_image is not None:
            visualize_video_frame_with_text(node.curr_image, text=target_name[:], resize=(224,224))
        else:
            time.sleep(0.5)

        is_stale = "img stale" in note

        fps = round(1.0 / (time.perf_counter() - t0))
        status = f"t={node.timestep:3} | Target: {target_name}"
        if note:
            status += f" [{note}]"
        status += f", {fps:3} smp/s"

        if is_stale and last_stale:
            pass  # suppress: only the first stale print matters
        elif target_name != last_target or (is_stale and not last_stale):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts}] {status}", end="", flush=True)
            last_target = target_name
        else:
            print(f"\r\033[K{status}", end="", flush=True)

        last_stale = is_stale

    rclpy.spin(node)

if __name__ == "__main__": 
    main()