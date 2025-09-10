#!/usr/bin/env bash
set -x

GPUS='0'
PORT=25502
GPUS_PER_NODE=1
CPUS_PER_TASK=6
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"
OUTPUT_DIR="./checkpoints/results/HCD_davis"

CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
  --output_dir=${OUTPUT_DIR} \
  --epochs 3 --lr_drop 6 8 \
  --dataset_file davis\
  --batch_size 1 \
  --pretrained_model_path ./weight
  
-