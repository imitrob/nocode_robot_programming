# nocode_robot_programming


## Notebooks

At `tests/`

0. `00_preparation.ipynb`: See the robot & gripper moving, see smooth motion
1. `01_teaching.ipynb`: Teaching via several modalities (gestures/joystick/kinesthetic)
2. `02a_state_decision_eval.ipynb`: Evaluation on `d1` dataset that was collected with previous notebook
3. `02b_AEGP_eval.ipynb`, `02b_AEGP_videoembedding.ipynb` Debug the AEGP model
4. `03_artificial_dataset_generator.ipynb`: See how the artificial dataset generation works.

## Tests (pytests)

1. `test_00_preparation.py`: Check if you can load modules

## Datasets 

D1: Collected dataset. Loader: `dataset_D1.py`
D2: Artificial dataset. Loader: `dataset_D2.py`