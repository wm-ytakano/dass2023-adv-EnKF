from typing import Tuple
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 水平
dx = 10.0  # 格子間隔
nx = 40  # 格子数
ny = 8  # 観測点数

# 時間
dt = 0.5  # 計算間隔[s]
nt = 1200  # 計算ステップ数
output_interval = 15  # 出力間隔
dt_out = dt * output_interval  # 出力間隔[s]
nt_out = nt // output_interval + 1  # 出力ステップ数
assim_interval = 15  # 同化間隔
assim_start = 375  # 同化開始ステップ
tt = 2  # 予報変数の時間次元 (leap-flogのため2時刻必要)


def forward(x: torch.Tensor, w: torch.Tensor, scheme: int) -> torch.Tensor:
    """
    input
        x: C with shape of (ens, tt, nx)
        w: w with shape of (nx,)
        scheme: leap frog=0 Euler=1
    return
        y: C with shape of (ens, tt, nx)
    """
    t = scheme
    u0 = 2.0  # 移流速度(定数)
    xnu = 5.0  # 拡散係数
    Cn = x[:, 1, :].roll(shifts=-1, dims=-1)
    Cp = x[:, 1, :].roll(shifts=1, dims=-1)
    A = -u0 * (Cn - Cp) / (2 * dx)
    Cn = x[:, t, :].roll(shifts=-1, dims=-1)
    Cc = x[:, t, :]
    Cp = x[:, t, :].roll(shifts=1, dims=-1)
    S = xnu * (Cn - 2 * Cc + Cp) / (dx * dx)
    x_next = torch.empty(x.shape)
    x_next[:, 0, :] = x[:, 1, :]
    x_next[:, 1, :] = x[:, t, :] + 2 * dt * (A + S + w.unsqueeze(0))
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
        x: C with shape of (nens, nx)
    output:
        y with shape of (nens, ny)
    """
    return x[:, 16:31:2]


def observation_true(x: torch.Tensor) -> torch.Tensor:
    """
    true observation oparator
    input
        x: C with shape of (nens, nx)
    output:
        y with shape of (nens, ny)
    """
    y_model = observation_model(x)
    noise = torch.normal(mean=0.0, std=8.0, size=y_model.shape)
    return y_model + noise


def run() -> (
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    x_true = torch.zeros((1, tt, nx))
    q_true = torch.normal(mean=0, std=1, size=(1,))
    x_true_list = []
    w_true_list = []
    y_list = []

    x_free = torch.zeros((1, tt, nx))
    x_free_list = []
    w_free_list = []
    q_free = torch.zeros((1,))  # no noise

    nens = 20
    x_asim = torch.zeros((nens, tt, nx))
    q_asim = torch.normal(mean=0, std=1, size=(nens,))
    x_asim_list = []
    w_asim_list = []

    for step in range(nt + 1):
        if step % 15 == 0:
            scheme = 1  # Euler
        else:
            scheme = 0  # leap frog
        w_true = forcing(step, q_true)
        x_true = forward(x_true, w_true, scheme)
        q_true = 0.8 * q_true + 0.2 * torch.normal(mean=0, std=1, size=(1,))

        w_free = forcing(step, q_free)
        x_free = forward(x_free, w_free, scheme)

        w_asim = forcing(step, q_asim)
        x_asim = forward(x_asim, w_asim, scheme)
        q_asim = 0.8 * q_asim + 0.2 * torch.normal(mean=0, std=1, size=(nens,))

        if step % assim_interval == 0 and step >= assim_start:
            x_f = x_asim[:, 1, :]  # (nens, nx)
            y = observation_true(x_true[:, 1, :])  # (nens, ny)
            increment = EnKF(x_f, y)  # (nens, nx)
            x_asim += increment.unsqueeze(1)  # (nens, tt, nx)
            y_list.append(y[0, :])  # (ny)

        if step % output_interval == 0:
            x_true_list.append(x_true[0, 1, :])
            w_true_list.append(w_true[0, :])
            x_free_list.append(x_free[0, 1, :])
            w_free_list.append(w_free[0, :])
            x_asim_list.append(x_asim[:, 1, :])
            w_asim_list.append(w_asim)

    return (
        torch.stack(x_true_list),  # (nt_out, nx)
        torch.stack(w_true_list),  # (nt_out, nx)
        torch.stack(y_list),  # (nt_out, ny)
        torch.stack(x_free_list),  # (nt_out, nx)
        torch.stack(w_free_list),  # (nt_out, nx)
        torch.stack(x_asim_list),  # (nt_out, nens, nx)
        torch.stack(w_asim_list),  # (nt_out, nens, nx)
    )


def EnKF(x_f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Ensemble Kalman Filter
    input:
        x_f: forecast with shape of (nens, nx)
        y: observation with shape of (nens, ny)
    output
        analysis increment with shape of (nens, nx)
    """
    nens = x_f.shape[0]
    x_mean = x_f.mean(dim=0).unsqueeze(0)  # (1, nx)
    delta_X = x_f - x_mean  # (nens, nx)
    delta_Y = observation_model(delta_X)  # (nens, ny)
    R = torch.diag(torch.full((ny,), 8.0**2))
    R2 = torch.matmul(delta_Y.t(), delta_Y) + (nens - 1) * R
    K = torch.matmul(
        torch.matmul(
            torch.linalg.inv(R2),
            delta_Y.t(),
        ),
        delta_X,
    )  # (ny, nx)
    Hx = observation_model(x_f)  # (nens, ny)
    # Perturbed observations method
    y_eps = torch.normal(mean=0.0, std=8.0, size=(nens, ny))
    y_inov = y + y_eps - Hx  # (nens, ny)
    increment = torch.matmul(y_inov, K)  # (nens, nx)
    return increment


def plot_xt(
    shade: torch.Tensor,
    shade_levels: torch.Tensor,
    contour: torch.Tensor,
    contour_levels: torch.Tensor,
) -> None:
    """
    plot data
    input:
        List of torch.Tensor with shape of (nt_out, nens, nx) (length of nt_out)
    """
    x_axis = torch.arange(0, dx * nx, dx)
    t_axis = torch.arange(0, dt_out * nt_out, dt_out)
    fig, ax = plt.subplots()
    mappable = ax.contourf(
        x_axis,
        t_axis,
        shade,
        cmap="RdBu_r",
        levels=shade_levels,
        extend="both",
    )
    fig.colorbar(mappable)
    ax.contour(
        x_axis,
        t_axis,
        contour,
        colors="black",
        levels=contour_levels,
        linewidths=0.5,
    )
    xmin = 160
    xmax = 300
    width = xmax - xmin
    tmin = assim_start * dt
    tmax = nt * dt
    height = tmax - tmin
    rect = patches.Rectangle(
        xy=(xmin, assim_start * dt),
        width=width,
        height=height,
        ec="#009900",
        fill=False,
    )
    ax.add_patch(rect)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("t [s]")
    ax.set_xlim([0, 400])
    ax.set_ylim([600, 0])
    plt.show()
