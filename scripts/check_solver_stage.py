import dataclasses
import numpy as np

from muav_isac import scenario as sc
from muav_isac.config import default_params
from muav_isac.bcd import _crb_feasible_beams
from muav_isac.beamforming import comm
from muav_isac.beamforming import sensing


def dbm_to_w(dbm):
    return 10 ** ((dbm - 30) / 10)


for pmax_dbm in [60,70,80]:

    print("\n")
    print("=" * 60)
    print(f"Pmax = {pmax_dbm} dBm")
    print("=" * 60)

    p = dataclasses.replace(
        default_params(),
        Pmax_W=dbm_to_w(pmax_dbm),
        N=16,
        T=60.0,
    )

    # 三种功率使用完全相同的随机网络
    net = sc.sample_network(
        p,
        np.random.default_rng(0)
    )

    traj = sc.init_trajectory(net)
    snap = sc.compute_channels(net, traj)

    # CRB可行的初始感知波束
    i0 = _crb_feasible_beams(net, snap)

    # ==========================================
    # 第一阶段：只测试通信波束
    # ==========================================

    print("\n>>> COMM START")

    G, g, comm_hist = comm.comm_beamforming(
        net,
        snap,
        i0,
        max_iter=1,
    )

    rate_after_comm = sc.sum_rate(
        net,
        snap,
        g,
        i0
    )

    print(">>> COMM END")
    print("comm history =", comm_hist)
    print("rate after comm =", rate_after_comm)
    comm_power = np.zeros((p.U, p.N))

    for u in range(p.U):
        for lv in range(net.Vu_list[u]):
            comm_power[u] += (
                    np.abs(g[(u, lv)]) ** 2
            ).sum(axis=1)

    sens_power = np.zeros((p.U, p.N))

    for u in range(p.U):
        for lt in range(net.Ku_list[u]):
            sens_power[u] += (
                    np.abs(i0[(u, lt)]) ** 2
            ).sum(axis=1)

    margin = p.Pmax_W - comm_power - sens_power

    print("Pmax =", p.Pmax_W)
    print("max comm power =", comm_power.max())
    print("max sensing power =", sens_power.max())
    print("minimum power margin =", margin.min())

    # ==========================================
    # 第二阶段：在上述通信波束基础上测试感知
    # ==========================================

    print("\n>>> SENSING START")

    I, i, sensing_hist = sensing.sensing_beamforming(
        net,
        snap,
        g,
        max_iter=1,
    )

    rate_final = sc.sum_rate(
        net,
        snap,
        g,
        i
    )

    print(">>> SENSING END")
    print("sensing history =", sensing_hist)
    print("final rate =", rate_final)