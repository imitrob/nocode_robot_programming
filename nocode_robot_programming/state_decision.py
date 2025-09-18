
import numpy as np

class StateDecision():
    def __init__(self):
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray): 
        '''
            X: shape (samples, w, h) 
            y: shape (samples, )
        '''
        self.model = None # Update

    def __call__(self, image: np.ndarray, timestep: float) -> tuple[bool, int]: # returns a branch (y) or -1 as anomaly
        return False, 0