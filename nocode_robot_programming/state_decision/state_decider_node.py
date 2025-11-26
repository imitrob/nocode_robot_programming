
import rclpy, time
from rclpy.node import Node
import argparse
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from cv_bridge import CvBridgeError, CvBridge

from video_embedding.utils import visualize_video_frame_with_text
from lfd_msgs.srv import StringService

from nocode_robot_programming.state_decision.dataset_auto import auto_load

from skills_manager.ros_utils import SpinningRosNode
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

import threading

WARNING_WHEN_IMAGE_OLDER_THAN = 0.2 # sec
MAX_TRAIN_TIME = 10.0 # sec

class StateDeciderNode(SpinningRosNode):
    def __init__(self, method: str):
        super(StateDeciderNode, self).__init__()

        if method == "SIFT":
            from nocode_robot_programming.state_decision.SIFT_model import StateDeciderSIFT
            self.model = StateDeciderSIFT()
        elif method == "DINO":
            from nocode_robot_programming.state_decision.dino_model import DINOFeaturePresence
            self.model = DINOFeaturePresence()
        elif method == "AEGP":
            from nocode_robot_programming.state_decision.AEGP_model import AEGP
            self.model = AEGP()
        elif method == "MANUAL":
            from nocode_robot_programming.state_decision.state_decider import StateDeciderManual
            self.model = StateDeciderManual()
        elif method == "BASE":
            from nocode_robot_programming.state_decision.state_decider import StateDeciderBase
            self.model = StateDeciderBase()
        else: raise Exception(f"Method '{method}' is not implemented!")

        
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

    def train_call(self, msg, res):
        self.task_name = msg.text
        self.training_finished.clear()
        self.training_request.set()
        if not self.training_finished.wait(MAX_TRAIN_TIME):
            raise Exception("Training not finished in time, adjust MAX_TRAIN_TIME")
        return res

    def train(self):
        assert self.task_name is not None

        dataset = auto_load(self.task_name)

        self.model.train(dataset.X, dataset.y_int, dataset.y_cls)
        time.sleep(5)

    def image_callback(self, msg):
        try:
            resized_img_gray = self.bridge.imgmsg_to_cv2(msg)
            self.curr_image = resized_img_gray.reshape((1, resized_img_gray.shape[0], resized_img_gray.shape[1]))
            self.timestep = int(msg.header.frame_id)
            self.curr_image_timestamp = time.time()
        except CvBridgeError as e:
            print(e)

    def predict(self):
        if self.curr_image is None:
            print("Not receiving image from '/modified_img' topic", flush=True)
            return "no img"
        if (time.time() - self.curr_image_timestamp) > WARNING_WHEN_IMAGE_OLDER_THAN:
            print(f"Image too old: {round(time.time() - self.curr_image_timestamp, 2)} sec", flush=True)

        target_name = self.model.predict(self.curr_image, self.timestep)
        print(f"{target_name=}")
        self.state_pub.publish(String(data=target_name))
        return target_name

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="State Decider Node")
    parser.add_argument('--name_method', type=str, help='SIFT/DINO/AEGP/BASE', choices=["SIFT", "DINO", "AEGP", "BASE"], default="BASE")
    args = parser.parse_args()

    rclpy.init()
    node = StateDeciderNode(args.name_method)
    
    while rclpy.ok(): # predict thread
        t0 = time.perf_counter()
        
        if node.training_request.is_set():
            node.training_request.clear()
            print("Training in progress", flush=True)
            node.train()
            node.training_finished.set()
            print("Training finished", flush=True)

        target_name = node.predict()
        if node.curr_image is not None:
            visualize_video_frame_with_text(node.curr_image, text=target_name[-8:])
        else:
            time.sleep(0.5)

        print(f"{round(1.0 / (time.perf_counter()-t0))} samples per second")

    rclpy.spin(node)