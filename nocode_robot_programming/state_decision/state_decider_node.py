
import rclpy, time
from rclpy.node import Node
import argparse
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from cv_bridge import CvBridgeError, CvBridge

from video_embedding.utils import visualize_video_frame_with_text

from skills_manager.ros_utils import SpinningRosNode

WARNING_WHEN_IMAGE_OLDER_THAN = 0.2 # sec

class StateDeciderNode(SpinningRosNode):
    def __init__(self, method: str):
        super(StateDeciderNode, self).__init__()

        if method == "SIFT":
            from nocode_robot_programming.state_decision.SIFT_model import StateDeciderSIFT
            self.model = StateDeciderSIFT()
        elif method == "DINO":
            from nocode_robot_programming.state_decision.dino_model import DINOStateDecider
            self.model = DINOStateDecider()
        elif method == "AEGP":
            from nocode_robot_programming.state_decision.AEGP_model import AEGP
            self.model = AEGP()
        elif method == "BASE":
            from nocode_robot_programming.state_decision.state_decider import StateDeciderBase
            self.model = StateDeciderBase()
        else: raise Exception(f"Method '{method}' is not implemented!")


        self.create_subscription(Image, "/modified_img", self.image_callback, 5)
        self.bridge = CvBridge()
        self.state_pub = self.create_publisher(String, "/target_state", 5)

        self.timestep = 0
        self.curr_image = None
        self.last_curr_image = 0.0

    def image_callback(self, msg):
        try:
            resized_img_gray = self.bridge.imgmsg_to_cv2(msg)
            self.curr_image = resized_img_gray.reshape((1, resized_img_gray.shape[0], resized_img_gray.shape[1]))
            self.timestep = int(msg.header.frame_id)
            self.last_curr_image = time.time()
        except CvBridgeError as e:
            print(e)

    def predict(self):
        if self.curr_image is None:
            print("Not receiving image from '/modified_img' topic", flush=True)
            return "no img"
        if (time.time() - self.last_curr_image) > WARNING_WHEN_IMAGE_OLDER_THAN:
            print(f"Image too old: {round(time.time() - self.last_curr_image, 2)} sec", flush=True)

        _, target_name = self.model.predict(self.curr_image, self.timestep)
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
        
        target_name = node.predict()
        if node.curr_image is not None:
            visualize_video_frame_with_text(node.curr_image, text=target_name[-8:])
        else:
            time.sleep(0.5)

        print(f"{round(1.0 / (time.perf_counter()-t0))} samples per second")

    rclpy.spin(node)