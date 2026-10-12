# 心脏 MRI 欠采样重建：U-Net 基线与 ISTA-Net+ 联合采样

## 毕业设计扩展（2026-10-12）

已完成固定/可学习 mask 三档实验及 K=3/6/9/12 深度消融。训练实际发生于
2026-10-10 至 2026-10-11，不能用任务书中的计划日期替代。

| 方法 | CS10 PSNR / SSIM | CS20 PSNR / SSIM | CS30 PSNR / SSIM |
|---|---|---|---|
| U-Net+DC（20轮） | 23.99 / 0.7578 | 28.45 / 0.8742 | 31.77 / 0.9264 |
| ISTA-Net+ 固定 mask（50轮） | 24.24 / 0.7736 | 30.92 / 0.9223 | 34.17 / 0.9587 |
| ISTA-Net+ 可学习 mask（50轮） | 27.24 / 0.8533 | 33.04 / 0.9496 | 36.62 / 0.9769 |

这里是全图逐片均值。机器可读报告、ROI病例指标、完整日志、模型和研究图在
[`results/graduation/`](results/graduation/)，主入口为
[`REPORT.md`](results/graduation/REPORT.md)。九组模型独立重评估全部通过；
`reevaluation.json` 保存逐指标及逐病例对照，MPS上最大差异为0。
发布的是末轮推理checkpoint，完整优化器续训状态另保留在本地毕业设计交付目录。

**解释边界：** U-Net与ISTA不是等训练预算比较。数据为MSD心脏幅值图的回顾性模拟，
不是原始多线圈k空间。历史调试观察过同一测试划分，属于探索性比较；只有单种子和6例测试。
学习mask前向严格保持19/38/58条线（包含ACS），STE概率均值不是实际采样率。
固定展示的la_005第72张切片在CS10上U-Net+DC优于ISTA，平均提升不表示逐片都改善。

```bash
# 原有数据预处理与U-Net基线见下文；使用项目虚拟环境
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/probe_ista.py --help

# 从零重训请使用新目录，避免覆盖公布结果；自动保存epoch状态
.venv/bin/python src/run_graduation.py --out results/new_run --device mps

# 用公布的九组模型重算指标，不训练；CPU也可用但更慢
.venv/bin/python src/eval_ista.py --results results/graduation --device mps

# 生成报告与图（还需要results/recon_cs10/20/30.pt三个U-Net权重）
.venv/bin/python src/report_ista.py --results results/graduation --device mps
```

`run_ista_ablation.py --kind hard-dc` 会**重新训练**有/无末尾DC的模型，属于可选新实验；
本论文采用 `report_ista.py` 中同一已训练checkpoint的**推理开关**消融，两者不能混用。

### 实现与复现记录

- `src/ista_net.py`：特征域D/F/阈值/逆F/G残差结构，K9有336,978参数。
- 所有频谱移位显式指定空间维。二维路径保存复数k空间；快速Cartesian路径保存复数伴随图，
  后者必须传 `fast_cartesian=True`，不能仅按张量形状猜测含义。
- 固定mask缓存测量；联合训练在autograd图中动态生成测量，ACS固定，外围top-k + sigmoid STE。
- `train_ista.py`按epoch保存优化器、调度器、随机状态及数据/源码指纹。恢复要求配置和指纹一致。
- CS10固定mask前15轮使用二维FFT，随后保留优化器迁移到等价一维实现。迁移记录在checkpoint
  和结果中。数学等价不意味着浮点轨迹逐比特一致；从零重训不承诺复现全部历史浮点位。
- `requirements.txt`为实际训练环境；文档依赖单列在`requirements-docs.txt`，不影响训练指纹。

### 论文工具

论文及学校模板保留在用户本地毕业设计目录，不公开学籍材料。
`fill_thesis_from_report.py`只更新五张数值表，可重复运行，不覆盖人工审阅的分析段落。
`build_thesis_docx.py`输出可编辑OMML公式与Word目录域；`export_thesis_lo.py`用实际PDF分页回填
目录缓存并验证稳定。macOS无界面导出使用`src/thesis_fonts.conf`加载本机Word和系统字体；
字体文件不随仓库分发。`check_thesis.py`核对正文至少15000汉字、引用、图表、公式编号、
页边距及Word/Markdown数值与实验JSON一致。跨设备重排后需再次更新目录。

以下为2026年8月至9月的U-Net与分割基线记录；分割并非本次ISTA论文的新增实验。

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
这两条趋势在本次口径变化下保留；是否能外推仍需独立数据验证。**绝对值只在本仓库这一套固定口径下可比**。

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
`mask_seed`，否则评估的是另一个采样模式。提交记录不能代替独立数值核验。

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
- 本节历史基线是纯图像域U-Net，mask固定；新增ISTA-Net+与联合mask实验见本文开头。
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
