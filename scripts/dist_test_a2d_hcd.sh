#!/usr/bin/env bash
set -x

GPUS='0'
PORT=25503
GPUS_PER_NODE=1
CPUS_PER_TASK=8
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

OUTPUT_DIR="./checkpoints/results/HCD_a2d"

CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
  --eval \
  --output_dir=${OUTPUT_DIR} \
  --dataset_file a2d \
  --batch_size 1 \
  --epochs 6 \
  --lr_drop 3 5 \
  --pretrained_model_path ./weight

