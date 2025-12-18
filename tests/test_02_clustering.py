
from nocode_robot_programming.state_decision_dataset_prepare.decision_state_clustering import cluster_branching_windows

def test_cluster_branching_windows():

    ds1 = 36
    ds2 = 45
    e = 10
    v = cluster_branching_windows([ds1, ds2], e)
    assert v['count'] == 1
    assert v['windows'] == [(ds1,ds2 + e)]

    ds1 = 36
    ds2 = 45
    e = 5

    v = cluster_branching_windows([ds1, ds2], e)
    assert v['count'] == 2
    assert v['windows'] == [(ds1,ds1+e), (ds2, ds2+e)]


if __name__ == "__main__":
    test_cluster_branching_windows()