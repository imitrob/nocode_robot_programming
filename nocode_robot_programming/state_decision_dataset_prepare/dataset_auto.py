import torch

from nocode_robot_programming.state_decision.utils import Filename
from nocode_robot_programming.state_decision_dataset_prepare.dataloader import ImageDatasetView, saved_img_processing
from nocode_robot_programming.state_decision_dataset_prepare.decision_state_clustering import cluster
from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset
from typing import List, Tuple
import numpy as np

DEBUG = False

class TrajectoryDatasetEvaluationViewBuilder(TrajectoryDataset):

    def report_average_times(self):
        def get_mean_time(modality: str, task: str):
            lst = [
                (self.tasks[t]['names'][0], np.array([
                    self.tasks[t]['lengths'][n] / 20.0
                    for n,tt in enumerate(self.tasks[t]['names']) 
                    if Filename(tt).is_demo == True
                ]).mean().round(1))
                for t in self.tasks.keys()
            ]

            avg = []
            for n, t in lst:
                if modality in n and task in n:
                    avg.append(t)

            if len(avg) == 0:
                mean = None
            else:
                mean = round(sum(avg) / len(avg), 1)
            return f"Task {task} captured with {modality} took on average: {mean}s"

        for m in ['kin', 'joy', 'gst']:
            for t in ['peg_pick', 'probe', 'wrap']:
                print(get_mean_time(m, t))

    def get_auto_dataset_view(self, file_names: list[str], relevant_parts: list[str], at: slice = slice(None,None), anomaly=False) -> ImageDatasetView:
        """ NO TAGS!    
            Fileloader used to create dataset
            1. Image preprocessing
            2. Mask only a portion of the samples
            3. Returns as a ImageDatasetView object
        """
        X_parts, Xt_parts, y_int_parts, y_name_parts = [], [], [], []

        start = at.start if at.start is not None else -float("inf")
        stop  = at.stop  if at.stop  is not None else  float("inf")

        if anomaly:
            offsets = [Filename(part).offset for part in relevant_parts]
            root_part = relevant_parts[np.argmin(offsets)]
            relevant_parts = [root_part]
        if DEBUG: print(f"{relevant_parts=}")

        for file in file_names:
            f = Filename(file)
            idx = self.files.index(f"{self.dir}/{file}.npz")
            imgs = saved_img_processing(self[idx]['img'].squeeze()).squeeze()   # (nsamples, H, W)
            if imgs.ndim == 2: # if nsmaples == 1, squeeze squeezes too hard, we need ndim=3
                imgs = imgs.unsqueeze(0)
            nsamples = imgs.shape[0]
            
            if anomaly:
                if f.part_name == root_part:
                    i = 0
                    label_name = f.part_name
                else:
                    i = 1
                    label_name = "anomaly"
            else:
                i = relevant_parts.index(f.part_name)
                label_name = f.part_name
            if DEBUG: print(f"{label_name=}")
            xt = torch.arange(f.offset, f.offset + nsamples)  # shape (nsamples,)
            
            mask = (xt >= start) & (xt < stop)                # safe, no OOB
            if mask.any():
                imgs_sub = imgs[mask]
                xt_sub   = xt[mask]

                y_int_sub   = torch.full((imgs_sub.shape[0],), i, dtype=torch.int)
                y_names_sub = [label_name] * imgs_sub.shape[0]

                X_parts.append(imgs_sub)
                Xt_parts.append(xt_sub)
                y_int_parts.append(y_int_sub)
                y_name_parts.extend(y_names_sub)

        if len(X_parts) == 0:
            return None

        X  = torch.cat(X_parts, dim=0)             # (total_samples, H, W)
        Xt = torch.cat(Xt_parts, dim=0)            # (total_samples,)
        y_int = torch.cat(y_int_parts, dim=0)      # (total_samples,)
        y_names = y_name_parts

        return ImageDatasetView(X=X, Xt=Xt, y_int=y_int, y_names=y_names, y_cls=relevant_parts)


    # def load_anom(self, task_name: str, e: int = 10) -> List[Tuple[ImageDatasetView, ImageDatasetView, str]]:
    #     return load_eval(self, task_name, e, anomaly=True)

    def load_eval_from_task(self, task_name: str, e: int = 10, anomaly: bool = False) -> List[Tuple[ImageDatasetView, ImageDatasetView, str]]:
        """ Returns list of dataset tuples, each tuple has train dataset, test dataset and text description. """
        decision_states = cluster(self.tasks[task_name], e)
        if DEBUG: print("Decision states: ", decision_states)

        index = self.tasks[task_name]

        datasets = []
        for ds in decision_states:

            file_names = []
            for name in index['names']:
                f = Filename(name)
                if f.part_name in ds['relevant_parts']:
                    file_names.append(name) 
            if DEBUG: print("filenames: ", file_names)

            # train/test split based on if it is a demonstration or not
            if anomaly:
                train_file_names, test_file_names = [], []
                for name in file_names:
                    f = Filename(name)
                    if f.offset == 0:
                        train_file_names.append(name)
                    else:
                        test_file_names.append(name)
            else:
                train_file_names, test_file_names = [], []
                for name in file_names:
                    f = Filename(name)
                    if f.is_demo or f.trial == 0:
                        train_file_names.append(name)
                    else:
                        test_file_names.append(name)


            d_train = self.get_auto_dataset_view(relevant_parts=ds['relevant_parts'], at=slice(ds['start'], ds['end']), file_names=train_file_names, anomaly=anomaly)
            d_test = self.get_auto_dataset_view(relevant_parts=ds['relevant_parts'], at=slice(ds['start'], ds['end']), file_names=test_file_names, anomaly=anomaly)
            
            
            if d_train is None:
                continue
            
            elif d_test is None:
                d_test = d_train

            f = Filename(ds['relevant_parts'][0])

            datasets.append((d_train, d_test, f"{f.task_userstudy} {f.modality} {f.person}, window={e}, anomaly={anomaly}, train={len(train_file_names)}, test={len(test_file_names)}"))


        return datasets


    def load_deploy(self, task_name: str, e: int = 10) -> tuple[list[ImageDatasetView], ImageDatasetView]:
        """ Returns list of dataset tuples, each tuple has train dataset, test dataset and text description. 
        
            - Each dataset tuple includes a single DS, exception is the last that contains all images.
        """

        # cluster decision states
        # [{'start': 8, 'end': 18, 'relevant_parts': ['ttst_kin_test', 'ttst_kin_test_branch_from_0_at_8']}, # DS1
        #   ... # DS2
        #   ... # DS3
        # ]
        decision_states = cluster(self.tasks[task_name], e)
        if DEBUG: print("Decision states: ", decision_states)

        index = self.tasks[task_name]

        datasets = []
        for ds in decision_states:

            file_names = []
            for name in index['names']:
                if Filename(name).part_name in ds['relevant_parts']:
                    file_names.append(name) 
            d = self.get_auto_dataset_view(self, relevant_parts=ds['relevant_parts'], at=slice(ds['start'], ds['end']), file_names=file_names)
            if d is not None:
                datasets.append(d)

        file_names = self.tasks[task_name]['names']
        relevant_parts = []
        for name in file_names:
            if Filename(name).is_demo:
                relevant_parts.append(name)

        all_images_dataset = self.get_auto_dataset_view(self, relevant_parts=relevant_parts, at=slice(None, None), file_names=file_names)

        return datasets, all_images_dataset

