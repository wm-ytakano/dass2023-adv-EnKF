from typing import List, Tuple
import torch
import matplotlib.pyplot as plt

# 水平
dx = 10.0  # 格子間隔
nx = 40  # 格子数
nobs = 8  # 格子点数

# 時間
dt = 0.5  # 計算間隔[s]
nt = 1200  # 計算ステップ数
output_interval = 15  # 出力ステップ間隔
dt_out = dt * output_interval  # 出力間隔[s]
nt_out = nt // output_interval + 1  # 出力ステップ数

tt = 2  # 予報変数の時間次元 (leap-flogのため2時刻必要)


def forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    input
        x: C with shape of (ens, tt, nx)
        w: w with shape of (nx,)
    return
        y: C with shape of (ens, nt, nx)
    """
    u0 = 2.0  # 移流速度(定数)
    xnu = 5.0  # 拡散係数
    Cn = x[:, 1, :].roll(shifts=-1, dims=-1)
    Cp = x[:, 1, :].roll(shifts=1, dims=-1)
    A = -u0 * (Cn - Cp) / (2 * dx)
    Cn = x[:, 0, :].roll(shifts=-1, dims=-1)
    Cc = x[:, 0, :]
    Cp = x[:, 0, :].roll(shifts=1, dims=-1)
    S = xnu * (Cn - 2 * Cc + Cp) / (dx * dx)
    x_next = torch.empty(x.shape)
    x_next[:, 0, :] = x[:, 1, :]
    x_next[:, 1, :] = x[:, 0, :] + 2 * dt * (A + S + w.unsqueeze(0))
    return x_next


def forcing(step: int, q: torch.Tensor) -> torch.Tensor:
    """
    input
        step: time step
        q: noise with shape of (nens,)
    return
        modeled forcing with shape of (nens, nx)
    """
    F0 = 1.0  # 強制項の振幅
    index = torch.arange(nx)
    nens = q.shape[0]
    index[6:] = 0
    k = torch.tensor(step, dtype=torch.float)
    tmp_f = torch.ones((nens, nx)) * torch.sin(2 * torch.pi * k * dt / 120)
    tmp_q = q.unsqueeze(1)
    return F0 * (tmp_f + tmp_q) * torch.sin(torch.pi * index * dx / 60)


def observation_model(x: torch.Tensor) -> torch.Tensor:
    """
    modeled observation oparator
    input
        x: C with shape of (nens, nt, nx)
    output
        y with shape of (nens, nobs)
    """
    return x[:, 1, 16:31:2]


def observation_true(x: torch.Tensor) -> torch.Tensor:
    """
    modeled observation oparator
    input
        x: C with shape of (nens, nt, nx)
    output:
        y with shape of (nens, nobs)
    """
    y_model = observation_model(x)
    noise = torch.normal(mean=0.0, std=8.0, size=y_model.shape)
    return y_model + noise


def run_free() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    model free run
    output
        list of C with shape of (nens, nx)
        list of w with shape of (nx,)
    """
    x_free = torch.zeros((1, tt, nx))
    x_free_list = []
    w_free_list = []
    q_free = torch.zeros((1,))  # no noise
    for step in range(nt + 1):
        w_free = forcing(step, q_free)
        x_free = forward(x_free, w_free)
        if step % output_interval == 0:
            x_free_list.append(x_free[:, 1, :])
            w_free_list.append(w_free)
    return x_free_list, w_free_list


def run_true() -> (
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor] | None]
):
    """
    create true data
    """
    x_true = torch.zeros((1, tt, nx))
    q_true = torch.normal(mean=0, std=1, size=(1,))
    x_true_list = []
    w_true_list = []
    y_list = []
    for step in range(nt + 1):
        w_true = forcing(step, q_true)
        x_true = forward(x_true, w_true)
        q_true = 0.8 * q_true + 0.2 * torch.normal(
            mean=0, std=1, size=(1,)
        )  # AR(1) model
        if step % output_interval == 0:
            x_true_list.append(x_true[:, 1, :])
            w_true_list.append(w_true)
            if step >= 375:
                y = observation_true(x_true)
                y_list.append(y)
            else:
                y_list.append(None)
    return x_true_list, w_true_list, y_list


def plot_xt(x_list: List[torch.Tensor], ens: int, levels: torch.Tensor) -> None:
    """
    plot data
    input:
        List of torch.Tensor with shape of (nens, nx) (length of nt_out)
    """
    x_axis = torch.arange(0, dx * nx, dx)
    t_axis = torch.arange(0, dt_out * nt_out, dt_out)
    fig, ax = plt.subplots()
    mappable = ax.contourf(
        x_axis,
        t_axis,
        torch.stack(x_list)[:, ens, :],
        cmap="RdBu_r",
        levels=levels,
        extend="both",
    )
    fig.colorbar(mappable)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("t [s]")
    ax.set_xlim([0, 400])
    ax.set_ylim([600, 0])
    plt.show()
