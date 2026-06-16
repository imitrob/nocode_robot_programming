import torch

class StateDeciderModelManager():
    def __init__(self, modelfactory):
        self.modelfactory = modelfactory
        self.models = []

    def train(self, datasets, all_dataset):
        self.models = []
        # Use per-decision-state datasets so each model fires only in its timestep window.
        # Fall back to all_dataset only when there are no branch points (linear task).
        training_sets = datasets if datasets else [all_dataset]
        print(f"Training {len(training_sets)} model(s)...", flush=True)
        for i, single_DS_dataset in enumerate(training_sets):
            model = self.modelfactory()
            model.train(X=single_DS_dataset.X, y=single_DS_dataset.y_int, y_cls=single_DS_dataset.y_cls)
            t = single_DS_dataset.timestep_range()
            self.models.append([model, t['min'], t['max']])
            print(f"  [{i+1}/{len(training_sets)}] {model} | active t=[{t['min']}, {t['max']}]", flush=True)

    def predict(self, image: torch.Tensor, timestep: float) -> tuple[str, str]:
        predictions = []
        for model, min, max in self.models:
            if min <= timestep <= max: # valid model for this timestep
                image = torch.tensor(image, dtype=torch.float32).cuda() / 255.0
                predictions.append(model.predict(image, timestep))

        if len(predictions) == 0:
            return "continue", f"no model for t={timestep}"
        elif len(predictions) > 1:
            return predictions[0], f"multiple models for t={timestep}"
        else:
            return predictions[0], ""

    def manual_predict(self, node, timestep, task_name, part_name) -> str:
        if len(self.models) > 0:
            return self.models[0][0].manual_predict(node, timestep, task_name, part_name)
        else:
            return "nomodel"