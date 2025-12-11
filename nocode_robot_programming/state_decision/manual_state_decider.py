
from nocode_robot_programming.state_decision.state_decider import StateDeciderBase
from nocode_robot_programming.state_decision.utils import Filename

import torch 

class StateDeciderManual(StateDeciderBase):
    def __init__(self):
        self.model = None
        self.y_cls = None

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls: list[str]): 
        '''
            X: shape (samples, w, h) 
            y: shape (samples, )
            y_cls: list[str] list of classes (labels) 
        '''
        self.ds = []
        for label in y_cls:
            if Filename(label).offset != 0:
                self.ds.append(Filename(label).offset)
        
        print("Manual Training initialized!")
        print("stopping on", self.ds)

        self.target_label = Filename(y_cls[0]).task # any y_cls class has root of the name as task
        self.y_cls = y_cls

    def predict(self, image: torch.Tensor, timestep: float | None = None) -> str: 
        ''' returns a target label class from y_cls (str) or "" for anomaly
        '''
        if timestep in self.ds:
            return "manual_choose"
        else:
            return "continue"
    
    def predict_many(self, X):
        return [self.predict(x) for x in X]

