import dataclasses
import numpy as np

from muav_isac import scenario as sc
from muav_isac import baselines
from muav_isac.config import default_params


def dbm_to_w(dbm):
    return 10 ** ((dbm - 30) / 10)


for pmax_dbm in [60, 70, 80]:

    p = dataclasses.replace(
        default_params(),
        Pmax_W=dbm_to_w(pmax_dbm),
        N=16,
        T=60.0,
    )

    # 每种功率都使用完全相同的用户、目标、NLoS随机信道
    net = sc.sample_network(
        p,
        np.random.default_rng(0)
    )

    traj = sc.init_trajectory(net)

    result = baselines.bfwot(
        net,
        traj,
        bf_iters=3,
        verbose=True,
    )

    print()
    print(f"===== {pmax_dbm} dBm =====")
    print("rate history =", result.rate_history)
    print("final rate   =", result.final_rate)
    print()