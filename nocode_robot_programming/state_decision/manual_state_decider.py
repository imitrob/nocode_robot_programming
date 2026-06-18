
from nocode_robot_programming.state_decision.state_decider import StateDeciderBase
from nocode_robot_programming.state_decision.utils import Filename
from nocode_robot_programming.state_decision_dataset_prepare.decision_state_clustering import cluster, search_for_parent

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
        self.ds_timesteps = []
        self.ds_parents = []
        for label in y_cls:
            if Filename(label).offset != 0:
                self.ds_timesteps.append(Filename(label).offset)
                self.ds_parents.append(Filename(label).parent_offset)

        print("Manual Training initialized!")
        print("stopping on", self.ds_timesteps)

        self.target_label = Filename(y_cls[0]).task # any y_cls class has root of the name as task
        self.y_cls = y_cls

    def predict(self, image: torch.Tensor, timestep: float | None = None) -> str: 
        ''' returns a target label class from y_cls (str) or "" for anomaly
        '''
        pass # manual predict is called

    def manual_predict(self, node, timestep: int, task_name: str, part_name: str) -> str:
        """ Special function """
        if timestep in self.ds_timesteps:
            parent_idx = self.ds_timesteps.index(timestep)

            parent_offset = self.ds_parents[parent_idx]

            parent_name = search_for_parent(node.dataset_builder.tasks[task_name], parent_offset)
            print("manual_predict, timestep: ", timestep, "part_name: ", part_name, " parent_offset: ", parent_name, parent_name == part_name)

            if parent_name != part_name:
                return "continue"

            ds = cluster(node.dataset_builder.tasks[task_name])

            options = None
            for ds_ in ds:
                if ds_['start'] <= timestep <= ds_['end']:
                    options = ds_['relevant_parts']
                    break
            if options is None:
                # No clustered decision-state window contains this timestep. Degrade to
                # "continue" instead of crashing the predictor (was a hard assert), and
                # log loudly so the mismatch is still visible.
                print(f"[manual_predict WARNING] timestep {timestep} not linked to any DS from ({ds}); continuing", flush=True)
                return "continue"
            return "manual_choose|" + "|".join(options)
        else:
            return "continue"
    
    def predict_many(self, X):
        return [self.predict(x) for x in X]

