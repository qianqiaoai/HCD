#!/usr/bin/env bash
set -x

GPUS='0,1,2,3,4,5,6,7'
PORT=25502
GPUS_PER_NODE=8
CPUS_PER_TASK=6
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"
OUTPUT_DIR="./checkpoints/results/HCD_youtube"

CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} /usr/miniconda3/envs/hcd/bin/torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
  --output_dir=${OUTPUT_DIR} \
  --epochs 9 --lr_drop 6 8 \
  --dataset_file ytvos\
  --batch_size 2 \
  --pretrained_model_path ./weight 
-