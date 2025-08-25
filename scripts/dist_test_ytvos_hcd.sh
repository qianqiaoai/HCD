#!/usr/bin/env bash
set -x
cd..

GPUS='0'
PORT=29500
GPUS_PER_NODE=1
CPUS_PER_TASK=8
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

OUTPUT_DIR="./checkpoints/results/VDIT_${BACKBONE}_eval"
CHECKPOINT="./checkpoints/checkpoint.pth"
python inference_ytvos.py --with_box_refine --binary \
  --eval \
  --ngpu=${GPUS_PER_NODE} \
  --output_dir=${OUTPUT_DIR} \
  --resume=${CHECKPOINT} \
  --amp \


