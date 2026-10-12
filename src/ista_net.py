"""ISTA-Net+ and a learnable Cartesian sampling mask for cardiac MRI.

The reconstruction path follows the corrected ISTA-Net+ implementation used in
the handover notes:

* data consistency is evaluated in complex k-space;
* every ``fftshift``/``ifftshift`` touches spatial dimensions only;
* the learned transform operates in feature space;
* the image residual branch starts near zero, so a fresh model behaves like a
  stable gradient-descent initializer.

``LearnableMask`` uses a hard top-k line selection in the forward pass and the
sigmoid probabilities in the backward pass (straight-through estimator). The
center ACS lines remain fixed and the soft sampling-rate penalty is exposed to
the training script.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 数据一致性（复数 k 空间；fftshift 约定与 sampling.py 一致）
# ---------------------------------------------------------------------------

def dc_gradient(
    x: torch.Tensor,
    k_meas: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """数据保真项梯度 Re(A^T(Ax − y))，A = M·FFT。

    对实值图像 x 的最小二乘 ‖M·FFT(x) − k_meas‖² 的梯度（取实部）。
    x:      [B,1,H,W] 实值
    k_meas: [B,2,H,W]（[:,0]=real, [:,1]=imag）
    mask:   [H,W] 二值
    返回    [B,1,H,W] 实值
    """
    Fx = torch.fft.fftshift(torch.fft.fft2(x[:, 0]), dim=(-2, -1))   # [B,H,W] 复数
    k_m = torch.complex(k_meas[:, 0], k_meas[:, 1])            # [B,H,W]
    m = mask.to(dtype=Fx.dtype)
    resid = m * Fx - k_m                                       # 已采样处非零，未采样处 k_m=0
    grad = torch.fft.ifft2(torch.fft.ifftshift(resid, dim=(-2, -1)))  # [B,H,W] 复数
    return grad.real.unsqueeze(1)                              # [B,1,H,W]


def hard_dc(
    x: torch.Tensor,
    k_meas: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """硬 DC 投影：已采样 k 空间行替换为实测值。返回幅值图。"""
    k_x = torch.fft.fftshift(torch.fft.fft2(x[:, 0]), dim=(-2, -1))
    k_m = torch.complex(k_meas[:, 0], k_meas[:, 1])
    m = mask.to(dtype=k_x.dtype)
    # k_meas has already been masked. Applying m twice changes the STE gradient.
    k_out = k_x * (1.0 - m) + k_m
    return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_out, dim=(-2, -1)))).unsqueeze(1)


def simulate_measurement(
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a zero-filled reconstruction and complex k-space measurement.

    Parameters
    ----------
    gt:
        Full-sampled magnitude images with shape ``[B, 1, H, W]``.
    mask:
        Cartesian line mask with shape ``[H, W]``. It may carry an STE graph.

    Returns
    -------
    zf:
        Zero-filled magnitude reconstruction, shape ``[B, 1, H, W]``.
    k_meas:
        Real/imaginary channels of the measured k-space, shape
        ``[B, 2, H, W]``.
    """
    k_full = torch.fft.fftshift(
        torch.fft.fft2(gt[:, 0]), dim=(-2, -1)
    )
    k_meas = k_full * mask.to(dtype=k_full.dtype)
    zf = torch.abs(
        torch.fft.ifft2(torch.fft.ifftshift(k_meas, dim=(-2, -1)))
    ).unsqueeze(1)
    return zf, torch.stack((k_meas.real, k_meas.imag), dim=1)


def simulate_cartesian_measurement(
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equivalent measurement representation using a one-dimensional FFT.

    The project samples complete columns of the 2-D Cartesian k-space. Since
    the mask is constant along the readout frequency axis, the normal operator
    ``F^{-1}(M * F(.))`` reduces exactly to a row-wise 1-D FFT operator. The
    returned complex tensor is the zero-filled adjoint image, which is enough
    for both the soft gradient step and hard projection.
    """
    line_mask = mask[0].to(dtype=gt.dtype)
    k_full = torch.fft.fftshift(
        torch.fft.fft(gt[:, 0], dim=-1), dim=-1
    )
    zf_complex = torch.fft.ifft(
        torch.fft.ifftshift(k_full * line_mask, dim=-1), dim=-1
    )
    return zf_complex.abs().unsqueeze(1), torch.stack(
        (zf_complex.real, zf_complex.imag), dim=1
    )


def cartesian_dc_gradient(
    x: torch.Tensor,
    zf_complex: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fast row-wise form of the complex Cartesian data-consistency gradient."""
    line_mask = mask[0].to(dtype=x.dtype)
    fx = torch.fft.fftshift(
        torch.fft.fft(x[:, 0], dim=-1), dim=-1
    ) * line_mask
    adjoint_x = torch.fft.ifft(
        torch.fft.ifftshift(fx, dim=-1), dim=-1
    )
    measured = torch.complex(zf_complex[:, 0], zf_complex[:, 1])
    return (adjoint_x - measured).real.unsqueeze(1)


def cartesian_hard_dc(
    x: torch.Tensor,
    zf_complex: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fast row-wise hard DC projection equivalent to :func:`hard_dc`."""
    line_mask = mask[0].to(dtype=x.dtype)
    fx = torch.fft.fftshift(
        torch.fft.fft(x[:, 0], dim=-1), dim=-1
    )
    measured = torch.complex(zf_complex[:, 0], zf_complex[:, 1])
    current_sampled = torch.fft.ifft(
        torch.fft.ifftshift(fx * line_mask, dim=-1), dim=-1
    )
    return (x[:, 0] - current_sampled + measured).abs().unsqueeze(1)


# ---------------------------------------------------------------------------
# 可学习稀疏变换（忠实复现官方 ISTA-Net+ 相位结构）
#
#   x_input = x − ρ·∇_data        数据一致性梯度步
#   x_D = D(x_input)              1→nf 升维到特征空间
#   F:  conv→relu→conv（nf→nf）   正变换
#   soft-threshold
#   F̃:  conv→relu→conv（nf→nf）   逆变换
#   x_G = G(F̃(soft(F(x_D))))     nf→1 投影回图像
#   x_pred = x_input + x_G        残差连接（G 初始化小 → 起步≈纯梯度下降）
#   对称约束：F̃(F(x_D)) ≈ x_D（在特征空间，不是图像空间）
#
# 关键：残差分支 x_G 初始接近 0，网络起步等价于纯梯度下降（已验证能超过零填充），
# 再从这个好起点学习；而不是在零填充上叠随机噪声（之前版本起步低于零填充的根因）。
# ---------------------------------------------------------------------------

def soft_threshold(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * F.relu(x.abs() - thr)


class ISTAPhase(nn.Module):
    """ISTA-Net+ 单个展开相位，参数独立（不跨相位共享）。"""

    def __init__(self, num_filters: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        nf = num_filters
        self._rho = nn.Parameter(torch.tensor(0.5))             # softplus(0.5)≈0.97
        self._thr = nn.Parameter(torch.tensor(-4.0))            # softplus(-4)≈0.018
        self.conv_D  = nn.Conv2d(1,  nf, kernel_size, padding=pad, bias=False)
        self.conv1_f = nn.Conv2d(nf, nf, kernel_size, padding=pad, bias=False)
        self.conv2_f = nn.Conv2d(nf, nf, kernel_size, padding=pad, bias=False)
        self.conv1_b = nn.Conv2d(nf, nf, kernel_size, padding=pad, bias=False)
        self.conv2_b = nn.Conv2d(nf, nf, kernel_size, padding=pad, bias=False)
        self.conv_G  = nn.Conv2d(nf, 1,  kernel_size, padding=pad, bias=False)
        for m in (self.conv_D, self.conv1_f, self.conv2_f, self.conv1_b, self.conv2_b):
            nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(self.conv_G.weight)
        with torch.no_grad():
            self.conv_G.weight.mul_(0.1)                        # 残差分支起步小

    @property
    def rho(self) -> torch.Tensor:
        return F.softplus(self._rho)

    @property
    def thr(self) -> torch.Tensor:
        return F.softplus(self._thr)

    def forward(
        self,
        x: torch.Tensor,
        k_meas: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_input = x - self.rho * dc_gradient(x, k_meas, mask)
        x_D = self.conv_D(x_input)
        x_f = self.conv2_f(F.relu(self.conv1_f(x_D)))
        x_s = soft_threshold(x_f, self.thr)
        x_b = self.conv2_b(F.relu(self.conv1_b(x_s)))
        x_pred = x_input + self.conv_G(x_b)
        # 对称约束：F̃(F(x_D)) ≈ x_D（特征空间）
        x_est = self.conv2_b(F.relu(self.conv1_b(x_f)))
        sym = x_est - x_D
        return x_pred, sym


class ISTANetPlus(nn.Module):
    """ISTA-Net+：K 个独立参数的展开 ISTA 相位。

    forward 返回 (重建图, 对称约束损失)。推理时忽略第二项即可。
    """

    def __init__(
        self,
        num_phases: int = 9,
        num_filters: int = 32,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.phases = nn.ModuleList(
            [ISTAPhase(num_filters, kernel_size) for _ in range(num_phases)]
        )

    def forward(
        self,
        zf: torch.Tensor,
        mask: torch.Tensor,
        k_meas: torch.Tensor,
        final_hard_dc: bool = True,
        fast_cartesian: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        zf:     [B,1,H,W] 零填充初始化
        mask:   [H,W] 二值采样 mask
        k_meas: [B,2,H,W] 复数测量 k 空间
        返回 (重建图[B,1,H,W] 值域[0,1], 对称约束损失标量)
        """
        x = zf
        sym_loss = x.new_zeros(())
        for phase in self.phases:
            if fast_cartesian:
                x_input = x - phase.rho * cartesian_dc_gradient(
                    x, k_meas, mask
                )
                x_D = phase.conv_D(x_input)
                x_f = phase.conv2_f(F.relu(phase.conv1_f(x_D)))
                x_s = soft_threshold(x_f, phase.thr)
                x_b = phase.conv2_b(F.relu(phase.conv1_b(x_s)))
                x = x_input + phase.conv_G(x_b)
                x_est = phase.conv2_b(F.relu(phase.conv1_b(x_f)))
                sym = x_est - x_D
            else:
                x, sym = phase(x, k_meas, mask)
            sym_loss = sym_loss + sym.pow(2).mean()
        if final_hard_dc:
            if fast_cartesian:
                x = cartesian_hard_dc(x, k_meas, mask)
            else:
                x = hard_dc(x, k_meas, mask)
        return x.clamp(0.0, 1.0), sym_loss


class LearnableMask(nn.Module):
    """Learn a fixed-budget one-dimensional Cartesian sampling pattern.

    The mask is defined over phase-encoding lines. The hard forward mask keeps
    exactly ``round(size * ratio)`` lines, including a fixed central ACS block.
    The backward pass uses sigmoid probabilities, which gives the reconstruction
    network a dense gradient with respect to every candidate line.
    """

    def __init__(
        self,
        size: int,
        ratio: float,
        init_mask: torch.Tensor,
        acs_frac: float = 0.25,
        init_contrast: float = 0.05,
    ) -> None:
        super().__init__()
        if init_mask.ndim != 2 or init_mask.shape != (size, size):
            raise ValueError(f"init_mask must have shape [{size}, {size}]")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        if not 0.0 < acs_frac <= 1.0:
            raise ValueError("acs_frac must be in (0, 1]")
        if not 0.0 < init_contrast < 1.0:
            raise ValueError("init_contrast must be in (0, 1)")

        self.size = int(size)
        self.ratio = float(ratio)
        self.target_lines = max(1, int(round(size * ratio)))
        n_acs = max(1, int(round(self.target_lines * acs_frac)))
        self.outer_budget = self.target_lines - n_acs
        center = size // 2
        acs_lo = center - n_acs // 2
        acs_hi = acs_lo + n_acs

        acs = torch.zeros(size, dtype=torch.bool)
        acs[acs_lo:acs_hi] = True
        self.register_buffer("acs_lines", acs)

        initial = init_mask.detach().cpu()
        if not torch.all((initial == 0) | (initial == 1)):
            raise ValueError("initial mask must be binary")
        if not torch.equal(initial, initial[:1].expand_as(initial)):
            raise ValueError("initial mask must contain complete Cartesian lines")
        init_lines = initial[0].bool()
        if int(init_lines.sum()) != self.target_lines or not init_lines[acs].all():
            raise ValueError("initial mask must match the sampling budget and ACS")
        init_lines = init_lines & ~acs
        # Preserve the initial hard pattern while starting at the soft budget.
        outer_rate = self.outer_budget / max(1, size - n_acs)
        p = torch.full((size,), outer_rate * (1 - init_contrast))
        p[init_lines] += init_contrast
        p = p.clamp(1e-6, 1 - 1e-6)
        p[acs] = 0.5
        self.logits = nn.Parameter(torch.logit(p))

    def probabilities(self) -> torch.Tensor:
        """Return the soft per-line sampling probabilities."""
        probs = torch.sigmoid(self.logits)
        return torch.where(self.acs_lines, torch.ones_like(probs), probs)

    def _hard_lines(self, probs: torch.Tensor) -> torch.Tensor:
        """Select exactly the target number of phase-encoding lines."""
        hard = self.acs_lines.clone()
        k = self.outer_budget
        if k > 0:
            scores = probs.masked_fill(self.acs_lines, float("-inf"))
            selected = torch.topk(scores, k=k, dim=0).indices
            hard[selected] = True
        return hard

    def forward(self) -> torch.Tensor:
        """Return the STE mask with shape ``[H, W]``."""
        probs = self.probabilities()
        hard = self._hard_lines(probs).to(dtype=probs.dtype)
        ste_lines = hard + (probs - probs.detach())
        return ste_lines.unsqueeze(0).expand(self.size, -1)

    def hard_mask(self) -> torch.Tensor:
        """Return a detached binary mask for validation and inference."""
        probs = self.probabilities()
        hard = self._hard_lines(probs).to(dtype=probs.dtype)
        return hard.unsqueeze(0).expand(self.size, -1)

    def rate_penalty(self) -> torch.Tensor:
        """Soft penalty for deviation from the requested sampling ratio."""
        return (self.probabilities().mean() - self.target_lines / self.size).pow(2)

    def selected_lines(self) -> torch.Tensor:
        """Return selected line indices for logging and visualization."""
        return torch.where(self._hard_lines(self.probabilities()))[0]

    def summary(self) -> dict:
        """Return JSON-friendly mask statistics."""
        device_probs = self.probabilities().detach()
        hard = self._hard_lines(device_probs).cpu()
        probs = device_probs.cpu()
        return {
            "target_ratio": self.ratio,
            "target_lines": self.target_lines,
            "hard_ratio": float(hard.float().mean()),
            "hard_lines": int(hard.sum()),
            "acs_lines": int(self.acs_lines.sum()),
            "selected_lines": [int(x) for x in torch.where(hard)[0]],
            "line_probabilities": [float(x) for x in probs],
        }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
