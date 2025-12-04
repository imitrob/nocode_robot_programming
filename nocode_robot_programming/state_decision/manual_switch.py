
from typing import List

def manual_switch_keyboard(options: List[str]):
    print("=" * 20, flush=True)
    print("Valid options:", flush=True)
    print(f"{list(enumerate(options))}", flush=True)

    while True:
        ret = input()
        try:
            if int(ret) < len(options):
                break
        except:
            print("try again", flush=True)
    target_name = options[int(ret)]
    print("Target name is: ", target_name, flush=True)
    print("=" * 20, flush=True)
    return target_name

def manual_switch_goal_points():
    pass
    # goal_points

    # eef_point

    # diff_points = goal_points - eef_point

    

    # relative_map = 