import torch

class StateDeciderModelManager():
    def __init__(self, modelfactory):
        self.modelfactory = modelfactory
        self.models = []

    def train(self, datasets, all_dataset):
        self.models = []
        for single_DS_dataset in [*datasets, all_dataset]:
            model = self.modelfactory()
            model.train(X=single_DS_dataset.X, y=single_DS_dataset.y_int, y_cls=single_DS_dataset.y_cls)
            self.models.append([
                model, 
                single_DS_dataset.timestep_range()['min'],
                single_DS_dataset.timestep_range()['max'],
            ])
            print(f"Model {model} trained on {single_DS_dataset}")

    def predict(self, image: torch.Tensor, timestep: float) -> str:

        predictions = []
        for model, min, max in self.models:
            if min <= timestep <= max: # valid model for this timestep
                print(f"Selected model {model} because: {min} <= {timestep} <= {max}")
                predictions.append(model.predict(image, timestep))
        
        if len(predictions) == 0:
            print(f"No model available for timestep {timestep}")
            return ""
        elif len(predictions) > 1:
            print(f"More models available for timestep {timestep}, returning the first one")
            return predictions[0]
        else:
            return predictions[0]
