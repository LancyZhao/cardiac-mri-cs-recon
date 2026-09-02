#!/usr/bin/env bash
# 一条命令复现全部结果：环境 -> 数据 -> 预处理 -> 两个实验 -> 出图出表 -> 口径对照
#
# 用法：bash repro.sh
# 已有的中间产物会被跳过（venv / 数据 / 切片），删掉对应目录即可强制重跑。
# 耗时参考：Apple M 系列 MPS 上重建三档约 1.5-2h，分割两组约 40min。
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
DATA_URL=https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar

echo "==> 1/6 环境（版本锁定于 requirements.txt）"
if [ ! -x "$PY" ]; then
  command -v uv >/dev/null || { echo "需要 uv：https://docs.astral.sh/uv/"; exit 1; }
  uv venv --python 3.12 .venv
  VIRTUAL_ENV=.venv uv pip install -r requirements.txt
fi
$PY -c "import torch,numpy,skimage,scipy,nibabel;print('deps ok')"

echo "==> 2/6 数据集（MSD Task02_Heart，公开直接下载）"
mkdir -p data
if [ ! -d data/Task02_Heart ]; then
  [ -f data/Task02_Heart.tar ] || curl -L -o data/Task02_Heart.tar "$DATA_URL"
  tar -xf data/Task02_Heart.tar -C data/
fi

echo "==> 3/6 预处理（.nii.gz -> 192x192 切片，按病例划分）"
[ -f data/slices/meta.json ] || $PY src/prep_data.py --size 192 --test-frac 0.3

echo "==> 4/6 实验1 重建（CS 10/20/30%，seed=0 mask_seed=0）"
$PY src/train_recon.py --ratios 0.1,0.2,0.3 --epochs 20

echo "==> 5/6 实验2 分割"
$PY src/train_seg.py --epochs 20

echo "==> 6/6 出图出表 + 口径对照（全图/ROI × 切片/病例）"
$PY src/viz.py
$PY src/eval_recon.py

echo
echo "完成。结果在 results/：recon_results.json、recon_metrics_calibers.json、"
echo "seg_results.json、REPORT.md、figures/"
