import torch
from nocode_robot_programming.state_decision_dataset_prepare.decision_state_clustering import cluster
from typing import List, Dict
# TODO:

class StateDeciderMultiModel():
    def __init__(self, modelfactory):
        self.modelfactory = modelfactory
        self.models = []
        self.models_range = []

        self.window_size = 30

    def train(self, X: torch.Tensor, Xt: torch.Tensor, y: torch.Tensor):
        
        unique_branches = list(set(y))
        ret: List[Dict[str, str | int]] = cluster(unique_branches, self.window_size)
        chunks = ret['windows']
        nchunks = ret['count']
        
        print("==============================")
        print(f"Total {nchunks} models created")
        print(f"- model windows: {chunks}")
        print("==============================")

        for ichunk in range(nchunks):
            start, end = chunks[ichunk]

            Xch = torch.tensor([])
            tch = torch.tensor([])
            ych = torch.tensor([])
            for x,t,y_ in zip(X, Xt, y):
                if start <= t <= end:
                    Xch.contatenate([Xch, x])
                    tch.contatenate([tch, t])
                    ych.contatenate([ych, y_])

            model = self.modelfactory()
            
            model.train(Xch, ych)

            self.models.append(model)
            self.models_range.append((start, end))

    def predict(self, image: torch.Tensor, timestep: float) -> tuple[bool, int]: # returns a branch (y) or -1 as anomaly
        selected_model = None
        
        for n, model_range in enumerate(self.models_range):
            if model_range[0] <= timestep <= model_range[1]:
                print(f"Selected model {n} because: {model_range[0]} <= {timestep} <= {model_range[1]}")
                selected_model = n
        
        return self.models[selected_model].predict(image, timestep)

