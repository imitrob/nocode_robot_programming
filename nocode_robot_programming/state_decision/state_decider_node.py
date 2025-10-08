
import rclpy, time
from rclpy.node import Node
import argparse
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

from video_embedding.utils import visualize_video_frame_with_text

WARNING_WHEN_IMAGE_OLDER_THAN = 0.2 # sec

class StateDeciderNode(Node):
    def __init__(self, method: str):
        super().__init__("state_decider_node")

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
        self.create_subscription(Int32, "/timestep", self.image_callback, 5)
        self.state_pub = self.create_publisher(String, "/target_state", 5)

        self.timestep = 0
        self.curr_image = None
        self.last_curr_image = 0.0

    def image_callback(self, msg):
        self.last_curr_image = time.time()
        self.curr_image = msg

    def timestep_callback(self, msg):
        self.timestep = int(msg.data)

    def predict(self):
        if self.curr_image is None:
            print("Not receiving image from '/modified_img' topic", flush=True)
            return "no img"
        if (time.time() - self.last_curr_image) > WARNING_WHEN_IMAGE_OLDER_THAN:
            print(f"Image too old: {(time.time() - self.last_curr_image)} sec", flush=True)

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
        target_name = node.predict()
        if node.curr_image is not None:
            visualize_video_frame_with_text(node.curr_image, text=target_name[-8:])
        else:
            time.sleep(1.0)
        rclpy.spin_once(node)

    rclpy.spin(node)