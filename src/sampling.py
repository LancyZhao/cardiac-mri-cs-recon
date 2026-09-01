"""K 空间欠采样与指标。

欠采样用 **笛卡尔一维变密度随机** mask：只在相位编码方向抽线，中心保留一段全采样
(ACS, auto-calibration signal)。这是临床 Cartesian 序列真正能省时间的方式——
省的是相位编码步数。二维随机点采样在 2D 序列里物理上不可实现，只是论文里的简化。

诚实边界：MSD 数据是已重建的幅值图，不是原始多线圈 k 空间。这里做的是
retrospective undersampling（对幅值图做 FFT 再抽样），没有真实相位与线圈信息，
因此绝对 PSNR 会比在 fastMRI 原始数据上测得的偏乐观，不能与论文数字直接对比。
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def cartesian_mask(size: int, ratio: float, acs_frac: float = 0.25, seed: int = 0) -> np.ndarray:
    """生成 (size, size) 的一维变密度随机 mask，采样率约为 ratio。

    ratio: 采样的相位编码线占比（0.1 = CS 10%）
    acs_frac: 采样预算里分给中心全采样区的比例
    """
    n_lines = max(1, int(round(size * ratio)))
    n_acs = max(1, int(round(n_lines * acs_frac)))
    n_rand = max(0, n_lines - n_acs)

    center = size // 2
    acs_lo = center - n_acs // 2
    lines = set(range(acs_lo, acs_lo + n_acs))

    # 变密度：离中心越近概率越高（MRI 能量集中在低频）
    idx = np.arange(size)
    p = np.exp(-0.5 * ((idx - center) / (size * 0.15)) ** 2)
    p[list(lines)] = 0.0
    if p.sum() > 0 and n_rand > 0:
        p = p / p.sum()
        rng = np.random.default_rng(seed)
        picked = rng.choice(size, size=min(n_rand, int((p > 0).sum())), replace=False, p=p)
        lines.update(picked.tolist())

    mask = np.zeros((size, size), dtype=np.float32)
    mask[:, sorted(lines)] = 1.0  # 沿相位编码方向(列)抽线
    return mask


def undersample(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """图像 -> k空间 -> 按 mask 抽样 -> 零填充逆变换回图像域（取幅值）。"""
    k = np.fft.fftshift(np.fft.fft2(img))
    k_us = k * mask
    zf = np.fft.ifft2(np.fft.ifftshift(k_us))
    return np.abs(zf).astype(np.float32)


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, test, data_range=1.0))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    return float(structural_similarity(ref, test, data_range=1.0))


def dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """二值 Dice。pred/target 为 0/1 数组。"""
    p, t = pred.astype(np.float32), target.astype(np.float32)
    inter = float((p * t).sum())
    return (2.0 * inter + eps) / (float(p.sum()) + float(t.sum()) + eps)


if __name__ == "__main__":
    for r in (0.1, 0.2, 0.3):
        m = cartesian_mask(192, r, seed=0)
        got = m[0].sum() / 192
        print(f"CS {int(r*100):2d}%  ->  {int(m[0].sum()):3d}/192 lines  (actual {got:.3f})")
