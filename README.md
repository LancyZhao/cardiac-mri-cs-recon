# 心脏 MRI 欠采样重建与分割复现实验

在公开数据集上复现压缩感知 MRI 重建与左心房分割，产出可核验的 PSNR/SSIM/Dice 指标。

数据集：[Medical Segmentation Decathlon](http://medicaldecathlon.com/) Task02_Heart（心脏 MRI，
左心房标注，20 例 `.nii.gz`，公开直接下载）。

## 结果

测试集 6 例 / 670 切片，**按病例**划分（train 14 例 / test 6 例），192×192。
完整数据见 [`results/REPORT.md`](results/REPORT.md)。

重建（PSNR / SSIM，参考为全采样图，**全图口径 · 按切片平均**）：

| CS 比率 | 采样线数 | 零填充 | U-Net | U-Net+DC | 零填充 SSIM | U-Net+DC SSIM |
|---|---|---|---|---|---|---|
| 10% | 19/192 | 20.26 dB | 23.63 | **23.99** | 0.6412 | **0.7578** |
| 20% | 38/192 | 23.99 dB | 27.55 | **28.45** | 0.7683 | **0.8742** |
| 30% | 58/192 | 27.28 dB | 30.51 | **31.77** | 0.8440 | **0.9264** |

两个值得注意的现象：U-Net 相对零填充的增益基本恒定（+3.2~3.6 dB），不随采样率变化；
而数据一致性投影的额外收益随采样率**递增**（+0.36 / +0.90 / +1.26 dB）——采样线越多，
投影能塞回去的真实测量值越多，且这部分收益不需要训练。

### 换成 ROI 口径 + 按病例聚合会怎样（重要）

上表的绝对值系统性偏高，因为全图里像素 `<0.05` 的近零背景**平均占 51.5%**（670 张
测试切片实测），那里 GT 与重建都≈0、误差≈0，把 `PSNR = 10log10(1/MSE)` 的分母稀释了。
`src/eval_recon.py` 用同一批 checkpoint 同时给出两种口径（`results/recon_metrics_calibers.json`）：

| CS | 方法 | 全图/切片 | 全图/病例 ± 例间 std | **ROI/病例 ± 例间 std** | ROI 代价 |
|---|---|---|---|---|---|
| 10% | 零填充 | 20.26 | 20.25 ± 0.54 | **17.61 ± 0.40** | −2.65 dB |
| 10% | U-Net+DC | 23.99 | 23.90 ± 2.12 | **21.20 ± 2.18** | −2.79 dB |
| 20% | 零填充 | 23.99 | 23.98 ± 0.49 | **21.39 ± 0.35** | −2.60 dB |
| 20% | U-Net+DC | 28.45 | 28.39 ± 1.58 | **25.71 ± 1.63** | −2.74 dB |
| 30% | 零填充 | 27.28 | 27.27 ± 0.55 | **24.76 ± 0.34** | −2.52 dB |
| 30% | U-Net+DC | 31.77 | 31.72 ± 1.25 | **29.05 ± 1.28** | −2.72 dB |

SSIM 受背景抬升更严重：CS30 的 U-Net+DC 从 0.9264 掉到 **0.8862**，CS10 的零填充
从 0.6412 掉到 **0.4503**（−0.19）——采样率越低、伪影越重，均匀零背景把 SSIM 抬得越多。

**结论层面真正重要的是：相对结论在换口径后全部保留。** U-Net 相对零填充的增益仍近乎恒定
（ROI 下 +3.22 / +3.39 / +3.03 dB），DC 的额外收益仍随采样率递增（ROI 下 +0.37 / +0.93 / +1.26 dB）。
所以可外推的是这两条趋势，**绝对值只在本仓库这一套固定口径下可比**。

关于不确定性：`recon_results.json` 里的 `psnr_std` 是**切片间**标准差，用它算置信区间会把
不确定性低估 **3.5~8.7 倍**——有效样本量是 6 例而不是 670 张切片（CS30 U-Net+DC：
按切片 SEM 0.064 dB vs 按病例 0.510 dB）。分割实验一开始就按病例聚合，重建实验这一项
是后补的，两者现在口径一致。

分割（Dice，按病例聚合）：

| 配置 | 损失 | Dice(病例) | Dice(前景切片) |
|---|---|---|---|
| `baseline` | BCE | **0.8900** ± 0.0382 | 0.7554 |
| `loss_dice` | BCE + Dice | 0.7622 ± 0.0936 | 0.5962 |

加 Dice loss 反而掉 0.13。推测原因是约半数切片没有前景，逐样本 Dice 在 target 全 0
时退化成只剩平滑项（`eps=1.0`），梯度变成噪声。**这是假设不是结论**，验证方法是把
Dice 项限制在有前景的切片上、或把 `eps` 降到 1e-6 量级后重跑。

## 复现

一条命令跑完（数据下载 → 预处理 → 两个实验 → 出图出表 → 口径对照）：

```bash
bash repro.sh
```

分步执行：

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt   # 版本全部锁定

curl -L -o data/Task02_Heart.tar \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar
tar -xf data/Task02_Heart.tar -C data/

.venv/bin/python src/prep_data.py --size 192 --test-frac 0.3   # 数据流水线
.venv/bin/python src/train_recon.py --ratios 0.1,0.2,0.3 --epochs 20   # 实验1 重建
.venv/bin/python src/train_seg.py --epochs 20                          # 实验2 分割
.venv/bin/python src/viz.py                                            # 出图出表
.venv/bin/python src/eval_recon.py            # 用已有 checkpoint 重算两种口径，不重训
```

结果写入 `results/`：`recon_results.json`、`recon_metrics_calibers.json`、
`seg_results.json`、`REPORT.md`、`figures/`。

随机性：`seed=0`、`mask_seed=0` 固定，单次运行。`eval_recon.py` 复用训练时的
`mask_seed`，否则评估的是另一个采样模式。**注意**：本仓库只有一次提交历史，
可信度来自"你可以自己重跑"，不是来自开发历史——这是刻意不掩饰的一点。

## 实验1：K 空间欠采样重建

MRI 慢在 K 空间要逐行采集，少采行就能少采集时间，代价是欠采样伪影。
在 CS 10% / 20% / 30% 三个采样率下比较三种重建：

| 方法 | 说明 |
|---|---|
| 零填充 IFFT | 未采样处填 0 直接逆变换，无学习的下界 |
| U-Net | 输入零填充图，残差输出，L1 损失 |
| U-Net + DC | 在网络输出上做数据一致性投影，把已测量的 k 空间行换回测量值 |

欠采样 mask 用**笛卡尔一维变密度随机**：只沿相位编码方向抽线，中心保留一段全采样
（ACS）。这是临床 2D 序列真正能省时间的方式——省的是相位编码步数。二维随机点采样
在 2D 序列里物理不可实现，只是部分论文的简化设定。

指标 PSNR / SSIM，以全采样图为参考。

## 实验2：左心房分割 A/B 对照

同一 U-Net 骨干、**同一 epoch 预算**下只改训练配置，看每一项各值多少 Dice：

| 配置 | 损失 | 优化器 | LR 策略 | 增强 | 本快照是否已跑 |
|---|---|---|---|---|---|
| `baseline` | BCE | Adam 1e-3 | 恒定 | 无 | ✅ |
| `loss_dice` | BCE + Dice | Adam 1e-3 | 恒定 | 无 | ✅ |
| `optim_sgd` | BCE + Dice | SGD 1e-2 + momentum | 恒定 | 无 | ✗ |
| `tuned` | BCE + Dice | Adam 1e-3 | cosine 退火 | 翻转 + 旋转 | ✗ |
| `tuned_bce` | BCE | Adam 1e-3 | cosine 退火 | 翻转 + 旋转 | ✗ |

`results/` 里目前只有 `baseline` 与 `loss_dice` 两组结果，其余配置代码已就绪但未运行。

给所有配置相同训练预算是刻意的：如果只给调优版更多 epoch，"调优有效"就成了训练
轮数带来的假象。

`tuned_bce` 的作用是**去混淆**：实测 `loss_dice` 会拖低 Dice，所以 `tuned`
（BCE+Dice + cosine + 增强）直接对比 `baseline` 时，分不清差异来自损失还是来自
增强与调度。把损失固定成 BCE 再看 cosine + 增强，才能单独归因。

Dice 按**病例**聚合（整例的交集与体积和）再对测试病例取平均，另附前景切片的逐片
Dice 作参考。

## 方法学边界

诚实记录这套实验测不到什么：

- MSD 提供的是**已重建的幅值图**，不是原始多线圈 k 空间。这里做 retrospective
  undersampling（对幅值图 FFT 后抽样），没有真实相位与线圈灵敏度信息。因此绝对
  PSNR 偏乐观，**不能与 fastMRI 原始数据上的论文数字直接比较**。偏乐观的来源按影响排序：
  1. **相位丢失**。幅值图是实数非负，其 k 空间满足共轭对称 `K(-k)=K*(k)`，被抽掉的
     相位编码行的信息在对称半边仍部分存在，零填充和网络都能隐式利用这个冗余。真实
     multi-coil 复数数据没有这个对称性——等价于名义 CS 10% 的**有效采样率更高**。
  2. **无采集噪声**。欠采样是纯无噪线性投影，参考图与欠采样图共享同一份噪声实现，
     噪声从随机干扰退化为确定性信号（网络可学，且 DC 投影会把抽到的行连噪声一起
     精确放回）。真实加速采集里每条线的噪声照在，并行成像还有 g-factor 噪声放大。
  3. **单通道假设**，无 sensitivity map、无 SENSE/GRAPPA 的 g-factor 惩罚。
  4. **anti-aliasing resize 发生在 FFT 之前**（`prep_data.py`），参考图高频已被低通削过
     一轮，而欠采样丢的主要就是高频。
  5. 1/99 分位截断 + 逐体归一压掉极端亮点这类高频源，并把背景硬钉在 0。
  6. retrospective 而非 prospective：无不同稳态、运动伪影、序列参数变化。

  偏悲观的只在算力/模型侧（20-25 epoch、7.76M 参数、单种子、纯图像域 U-Net、mask 未
  联合优化）。**方法学侧几乎清一色偏乐观**，所以与申报书 KPI 的差距只会比表面更大。
- 指标口径本身会显著改变绝对值，本仓库两种都报（见上文）：全图口径含 51.5% 近零背景，
  ROI 口径低 2.5~2.8 dB；SSIM 差得更多。`data_range` 写死 `1.0` 这一项**实测只值
  0.03 dB**（670 张测试切片逐片 `max` 的中位数为 1.0000，`≥0.99` 占 97.2%），既不是水分
  也不是保守项。SSIM 用 skimage 默认 `win_size=7, gaussian_weights=False`，未对齐
  Wang 原论文常用参数。
- 训练/测试按**病例级**划分。同一病例相邻切片高度相似，按切片随机划分会让测试集
  泄漏进训练集，指标虚高。
- 单次运行、单一随机种子，未做多种子重复。例间 std 已报，但 6 例的样本量下它仍是
  粗估；换 `mask_seed` 会改变结果，单个 mask 的数字不能当作该 CS 比率的代表值。
- 重建网络是纯图像域 U-Net，不是展开式迭代网络（ISTA-Net / PUERT 那一类），也没有
  联合优化采样轨迹——mask 在这里是固定先验，不参与学习。
- ROI mask 由 GT 阈值 + 形态学估出（`sampling.body_mask`），不是解剖学标注；它只依赖
  GT、不依赖被评估的重建结果，避免"用预测结果定义评估区域"。masked-SSIM 的窗口在
  边界处仍会混入背景像素，这个残留偏差没有消除。

## 结构

```
src/prep_data.py    .nii.gz -> 归一化 2D 切片，按病例划分
src/sampling.py     笛卡尔欠采样 mask、PSNR/SSIM/Dice、ROI(体内)口径指标
src/unet.py         U-Net（重建用残差输出，分割用 logits 输出）
src/train_recon.py  实验1
src/train_seg.py    实验2
src/eval_recon.py   用已有 checkpoint 重算两种口径 × 两种聚合，不重新训练
src/viz.py          mask 图、loss 曲线、定性对比图、Markdown 指标表
repro.sh            一条命令走完全流程
```
