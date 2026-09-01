# 心脏 MRI 欠采样重建与分割复现实验

在公开数据集上复现压缩感知 MRI 重建与左心房分割，产出可核验的 PSNR/SSIM/Dice 指标。

数据集：[Medical Segmentation Decathlon](http://medicaldecathlon.com/) Task02_Heart（心脏 MRI，
左心房标注，20 例 `.nii.gz`，公开直接下载）。

## 结果

测试集 6 例 / 670 切片，**按病例**划分（train 14 例 / test 6 例），192×192。
完整数据见 [`results/REPORT.md`](results/REPORT.md)。

重建（PSNR / SSIM，参考为全采样图）：

| CS 比率 | 采样线数 | 零填充 | U-Net | U-Net+DC | 零填充 SSIM | U-Net+DC SSIM |
|---|---|---|---|---|---|---|
| 10% | 19/192 | 20.26 dB | 23.63 | **23.99** | 0.6412 | **0.7578** |
| 20% | 38/192 | 23.99 dB | 27.55 | **28.45** | 0.7683 | **0.8742** |
| 30% | 58/192 | 27.28 dB | 30.51 | **31.77** | 0.8440 | **0.9264** |

两个值得注意的现象：U-Net 相对零填充的增益基本恒定（+3.2~3.6 dB），不随采样率变化；
而数据一致性投影的额外收益随采样率**递增**（+0.36 / +0.90 / +1.26 dB）——采样线越多，
投影能塞回去的真实测量值越多，且这部分收益不需要训练。

分割（Dice，按病例聚合）：

| 配置 | 损失 | Dice(病例) | Dice(前景切片) |
|---|---|---|---|
| `baseline` | BCE | **0.8900** ± 0.0382 | 0.7554 |
| `loss_dice` | BCE + Dice | 0.7622 ± 0.0936 | 0.5962 |

加 Dice loss 反而掉 0.13。推测原因是约半数切片没有前景，逐样本 Dice 在 target 全 0
时退化成只剩平滑项（`eps=1.0`），梯度变成噪声。**这是假设不是结论**，验证方法是把
Dice 项限制在有前景的切片上、或把 `eps` 降到 1e-6 量级后重跑。

## 复现

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch numpy nibabel scikit-image scipy matplotlib tqdm

curl -L -o data/Task02_Heart.tar \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar
tar -xf data/Task02_Heart.tar -C data/

.venv/bin/python src/prep_data.py --size 192 --test-frac 0.3   # 数据流水线
.venv/bin/python src/train_recon.py --ratios 0.1,0.2,0.3 --epochs 20   # 实验1 重建
.venv/bin/python src/train_seg.py --epochs 20                          # 实验2 分割
.venv/bin/python src/viz.py                                            # 出图出表
```

结果写入 `results/`：`recon_results.json`、`seg_results.json`、`REPORT.md`、`figures/`。

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
  PSNR 偏乐观，**不能与 fastMRI 原始数据上的论文数字直接比较**。
- 训练/测试按**病例级**划分。同一病例相邻切片高度相似，按切片随机划分会让测试集
  泄漏进训练集，指标虚高。
- 单次运行、单一随机种子，未做多种子重复，未估计指标的置信区间。
- 重建网络是纯图像域 U-Net，不是展开式迭代网络（ISTA-Net / PUERT 那一类），也没有
  联合优化采样轨迹——mask 在这里是固定先验，不参与学习。

## 结构

```
src/prep_data.py    .nii.gz -> 归一化 2D 切片，按病例划分
src/sampling.py     笛卡尔欠采样 mask、PSNR/SSIM/Dice
src/unet.py         U-Net（重建用残差输出，分割用 logits 输出）
src/train_recon.py  实验1
src/train_seg.py    实验2
src/viz.py          mask 图、loss 曲线、定性对比图、Markdown 指标表
```
