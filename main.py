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


def forward(x: torch.Tensor, w: torch.Tensor, scheme: int) -> torch.Tensor:
    """
    input
        x: C with shape of (nens, nlag, nx)
        w: w with shape of (nx,)
        scheme: leap-frog=-2 Euler=-1
    return
        y: C with shape of (nens, nlag, nx)
    """
    t = scheme
    u0 = 2.0  # 移流速度(定数)
    xnu = 5.0  # 拡散係数
    Cn = x[:, -1, :].roll(shifts=-1, dims=-1)
    Cp = x[:, -1, :].roll(shifts=1, dims=-1)
    A = -u0 * (Cn - Cp) / (2 * dx)
    Cn = x[:, t, :].roll(shifts=-1, dims=-1)
    Cc = x[:, t, :]
    Cp = x[:, t, :].roll(shifts=1, dims=-1)
    S = xnu * (Cn - 2 * Cc + Cp) / (dx * dx)
    x_next = torch.empty(x.shape)
    x_next[:, 0:-1, :] = x[:, 1:, :]
    x_next[:, -1, :] = x[:, t, :] + 2 * dt * (A + S + w.unsqueeze(0))
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
        x: C with shape of (nens, nlag, nx)
    output:
        y with shape of (nens, ny)
    """
    return x[:, -1, 16:31:2]


def observation_true(x: torch.Tensor) -> torch.Tensor:
    """
    true observation oparator
    input
        x: C with shape of (nens, nlag, nx)
    output:
        y with shape of (nens, ny)
    """
    y_model = observation_model(x)  # (nens, ny)
    noise = torch.normal(mean=0.0, std=8.0, size=y_model.shape)
    return y_model + noise


def run(
    nlag: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    input:
        nlag: 予報変数の時間次元 (leap-flogのため2時刻以上、固定ラグスムーザーの場合は長くする)
    """
    x_true = torch.zeros((1, nlag, nx))
    q_true = torch.normal(mean=0, std=1, size=(1,))
    x_true_save = torch.zeros((nt + 1, nx))
    w_true_list = []
    y_list = []

    x_free = torch.zeros((1, nlag, nx))
    x_free_save = torch.zeros((nt + 1, nx))
    w_free_list = []
    q_free = torch.zeros((1,))  # no noise

    nens = 20
    x_asim = torch.zeros((nens, nlag, nx))
    q_asim = torch.normal(mean=0, std=1, size=(nens,))
    x_asim_save = torch.zeros((nt + 1, nx))
    w_asim_list = []

    for step in range(nt + 1):
        if step % 15 == 0:
            scheme = -1  # Euler
        else:
            scheme = -2  # leap frog
        w_true = forcing(step, q_true)
        x_true = forward(x_true, w_true, scheme)
        q_true = 0.8 * q_true + 0.2 * torch.normal(mean=0, std=1, size=(1,))

        w_free = forcing(step, q_free)
        x_free = forward(x_free, w_free, scheme)

        w_asim = forcing(step, q_asim)
        x_asim = forward(x_asim, w_asim, scheme)
        q_asim = 0.8 * q_asim + 0.2 * torch.normal(mean=0, std=1, size=(nens,))

        if step % assim_interval == 0 and step >= assim_start:
            y = observation_true(x_true)  # (nens, ny)
            increment = EnKF(x_asim, y)  # (nens, nlag, nx)
            x_asim += increment  # (nens, nlag, nx)
            y_list.append(y[0, :])  # (1, ny) -> (ny)

        if step - nlag < 0:
            x_true_save[0:step, :] = x_true[0, nlag - step :, :]
            x_free_save[0:step, :] = x_free[0, nlag - step :, :]
            x_asim_save[0:step, :] = x_asim[:, nlag - step :, :].mean(dim=0)
        else:
            x_true_save[step - nlag : step, :] = x_true[0, :, :]
            x_free_save[step - nlag : step, :] = x_free[0, :, :]
            x_asim_save[step - nlag : step, :] = x_asim[:, :, :].mean(dim=0)
        if step % output_interval == 0:
            w_true_list.append(w_true[0, :])
            w_free_list.append(w_free[0, :])
            w_asim_list.append(w_asim)

    return (
        x_true_save[::output_interval, :],  # (nt_out, nx)
        torch.stack(w_true_list),  # (nt_out, nx)
        torch.stack(y_list),  # (nt_out, ny)
        x_free_save[::output_interval, :],  # (nt_out, nx)
        torch.stack(w_free_list),  # (nt_out, nx)
        x_asim_save[::output_interval, :],  # (nt_out, nx)
        torch.stack(w_asim_list),  # (nt_out, nens, nx)
    )


def EnKF(x_f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Ensemble Kalman Filter
    input:
        x_f: forecast with shape of (nens, nlag, nx)
        y: observation with shape of (nens, ny)
    output
        analysis increment with shape of (nens, nlag, nx)
    """
    nens = x_f.shape[0]
    x_mean = x_f.mean(dim=0).unsqueeze(0)  # (1, nlag, nx)
    delta_X = x_f - x_mean  # (nens, nlag, nx)
    delta_Y = observation_model(delta_X)  # (nens, ny)
    R = torch.diag(torch.full((ny,), 8.0**2))  # (ny, ny)
    R2 = torch.matmul(delta_Y.t(), delta_Y) + (nens - 1) * R  # (ny, ny)
    K = torch.einsum(
        "ij,jkl->ikl",
        torch.matmul(
            torch.linalg.inv(R2),
            delta_Y.t(),
        ),
        delta_X,
    )  # (ny, nlag, nx)
    Hx = observation_model(x_f)  # (nens, ny)
    # Perturbed observations method
    y_eps = torch.normal(mean=0.0, std=8.0, size=(nens, ny))
    y_inov = y + y_eps - Hx  # (nens, ny)
    increment = torch.einsum("ij,jkl->ikl", y_inov, K)  # (nens, nlag, nx)
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
