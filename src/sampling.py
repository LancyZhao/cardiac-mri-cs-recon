"""K 空间欠采样与指标。

欠采样用 **笛卡尔一维变密度随机** mask：只在相位编码方向抽线，中心保留一段全采样
(ACS, auto-calibration signal)。这是临床 Cartesian 序列真正能省时间的方式——
省的是相位编码步数。二维随机点采样在 2D 序列里物理上不可实现，只是论文里的简化。

诚实边界：MSD 数据是已重建的幅值图，不是原始多线圈 k 空间。这里做的是
retrospective undersampling（对幅值图做 FFT 再抽样），没有真实相位与线圈信息，
因此绝对 PSNR 会比在 fastMRI 原始数据上测得的偏乐观，不能与论文数字直接对比。

偏乐观的具体来源（按影响排序，全部指向同一方向）：
1. 相位丢失。幅值图是实数非负，其 k 空间满足共轭对称 K(-k)=K*(k)，被抽掉的
   相位编码行的信息在对称半边仍部分存在，零填充与网络都能隐式利用这个冗余。
   真实 multi-coil 复数数据没有这个对称性 —— 等价于名义 CS 10% 的有效采样率更高。
2. 无采集噪声。欠采样是纯无噪线性投影，参考图与欠采样图共享同一份噪声实现，
   噪声从随机干扰退化为确定性信号（网络可学，且 data_consistency 会把抽到的行
   连噪声一起精确放回）。真实加速采集里每条线的噪声照在，并行成像还有 g-factor
   噪声随加速倍数放大。
3. 单通道假设。无 sensitivity map、无 SENSE/GRAPPA 的 g-factor 惩罚。
4. anti-aliasing resize 发生在 FFT 之前（prep_data.py），参考图高频已被低通削过
   一轮，而欠采样丢的主要就是高频。
5. 1/99 分位截断 + 逐体归一压掉极端亮点这类高频源，并把背景硬钉在 0。
6. retrospective 而非 prospective：无不同稳态、运动伪影、序列参数变化。

偏悲观的只在算力/模型侧（20-25 epoch、7.76M 参数、单种子、纯图像域 U-Net、
mask 未联合优化），方法学侧几乎清一色偏乐观。
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


# --------------------------------------------------------------------------
# ROI(体内)口径的指标
#
# 为什么需要：全图口径下近零背景占了测试集平均 51.5% 的像素（<0.05，670 张切片实测），
# 而背景处 GT 与重建都≈0、误差≈0。PSNR = 10log10(1/MSE) 的分母被这些
# "两边都对"的像素稀释，SSIM 在均匀零区的局部窗口里更是≈1。仅剔除背景就让
# MSE 放大 1/(1-0.515)=2.06x，对应 PSNR 下降约 3.1 dB（下界）。
# 所以全图口径的绝对值系统性偏乐观，审稿人通常要求在 ROI 内报指标。
#
# 这里两种口径都算、都报，不替换原口径——已公开的数字必须仍能对上。
# --------------------------------------------------------------------------


def body_mask(gt: np.ndarray, thr: float = 0.05) -> np.ndarray:
    """从 GT 估体内区域：阈值 -> 闭运算 -> 填孔 -> 保留最大连通域。

    规则刻意简单且只依赖 GT（与被评估的重建结果无关），避免"用预测结果定义
    评估区域"这种会自我美化的做法。
    """
    from scipy import ndimage

    m = gt > thr
    if not m.any():
        return np.ones_like(gt, dtype=bool)
    m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:  # 只留最大连通域，去掉孤立噪点
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m.astype(bool)


def psnr_masked(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> float:
    """只在 mask 内算 MSE 的 PSNR，data_range 仍为 1.0（与全图口径可比）。"""
    if not mask.any():
        return float("nan")
    mse = float(np.mean((ref[mask].astype(np.float64) - test[mask].astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim_masked(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> float:
    """全图算 SSIM map，再只在 mask 内求均值（skimage 不支持传 mask）。

    注意窗口跨越边界处会混入背景像素，这是 masked-SSIM 的固有局限，
    比"把整片零背景都算进均值"小得多，但不为零。
    """
    if not mask.any():
        return float("nan")
    _, smap = structural_similarity(ref, test, data_range=1.0, full=True)
    return float(np.mean(smap[mask]))


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
