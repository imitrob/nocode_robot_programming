import torch

class StateDeciderModelManager():
    def __init__(self, modelfactory, anomaly=False):
        self.modelfactory = modelfactory
        self.models = []

        if anomaly:
            self.percentile_keep = 0.1
        else:
            self.percentile_keep = None

    def train(self, datasets, all_dataset):
        self.models = []
        
        print(f"Training {len(datasets)} model(s)...", flush=True)
        for i, single_DS_dataset in enumerate(datasets):
            model = self.modelfactory(percentile_keep=self.percentile_keep)
            model.train(X=single_DS_dataset.X, y=single_DS_dataset.y_int, y_cls=single_DS_dataset.y_cls)
            t = single_DS_dataset.timestep_range()
            self.models.append([model, t['min'], t['max']])
            print(f"  [{i+1}/{len(datasets)}] {model} | active t=[{t['min']}, {t['max']}]", flush=True)

        if all_dataset and all_dataset.n > 0:
            single_DS_dataset = all_dataset
            model = self.modelfactory(percentile_keep=self.percentile_keep)
            model.train(X=single_DS_dataset.X, y=single_DS_dataset.y_int, y_cls=single_DS_dataset.y_cls)
            t = single_DS_dataset.timestep_range()
            self.models.append([model, t['min'], t['max']])
            print(f"  [anomaly] {model} | active t=[{t['min']}, {t['max']}]", flush=True)

    def predict(self, image: torch.Tensor, timestep: float) -> tuple[str, str]:
        """ Run every model whose timestep window covers `timestep`, in self.models order.

        Anomaly OFF: only per-context (decision-state) models exist.
            - no model active  -> "continue"
            - more than one    -> use the first (highest priority)
        Anomaly ON: the anomaly model spans the full timestep range and is added last,
            so there is always at least one active model (predictions never fall back to
            "continue"), and being last it has the lowest priority -> the decision-state
            model is always preferred when one is active.
        """
        predictions = []
        fired = []
        img = torch.tensor(image, dtype=torch.float32).cuda() / 255.0
        for model, min, max in self.models:
            if min <= timestep <= max: # valid model for this timestep
                predictions.append(model.predict(img, timestep))
                fired.append(model)

        if len(predictions) == 0:
            return "continue", f"no model for t={timestep}"

        # Diagnostics: surface the firing model's top class scores (the decision margin).
        scores = getattr(fired[0], "last_scores", None)
        note = " ".join(f"{n}={s:.3f}" for n, s in scores) if scores else ""
        if len(predictions) > 1:
            note = f"{note} | multiple models for t={timestep}" if note else f"multiple models for t={timestep}"
        return predictions[0], note

    def manual_predict(self, node, timestep, task_name, part_name) -> str:
        if len(self.models) > 0:
            return self.models[0][0].manual_predict(node, timestep, task_name, part_name)
        else:
            # No decision-state model (and anomaly off) -> proceed, same as predict()'s empty case.
            return "continue"