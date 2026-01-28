
import torch

from nocode_robot_programming.state_decision.dino_model import DINOFeaturePresence, DINOFeaturePresenceConcat, DINOFeaturePresenceAttnGated
from nocode_robot_programming.state_decision.dino_with_mil import DINOWithMIL
from nocode_robot_programming.state_decision.SIFT_model import StateDeciderSIFT
from nocode_robot_programming.state_decision.AEGP_model import AEGP

__all__ = ['DINOFeaturePresence', 'DINOFeaturePresenceConcat', 'DINOFeaturePresenceAttnGated', 'DINOWithMIL', 'StateDeciderSIFT', 'AEGP']

class StateDeciderBase():
    def __init__(self):
        self.model = None
        self.y_cls = None

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls: list[str]): 
        '''
            X: shape (samples, w, h) 
            y: shape (samples, )
            y_cls: list[str] list of classes (labels) 
        '''
        target_label = y[0]
        self.model = target_label
        self.y_cls = y_cls

    def predict(self, image: torch.Tensor, timestep: float | None = None) -> str: 
        ''' returns a target label class from y_cls (str) or "" for anomaly
        '''
        return "test"
    
    def predict_many(self, X):
        return [self.predict(x) for x in X]

