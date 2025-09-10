"""
Training script of HCD
Modified from DETR, SgMg and VD-IT (https://github.com/facebookresearch/detr, https://github.com/bo-miao/SgMg, https://github.com/buxiangzhiren/VD-IT) 
"""
import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path
import os
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import torch.amp as amp
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch, evaluate, evaluate_a2d
from models import build_model_diff_cross as build_model
from util.logger import TensorboardLogger
import opts
import warnings

warnings.filterwarnings("ignore")


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def main(args):

    utils.init_distributed_mode(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "configs"), 'w') as f:
        f.write(str(args) + '\n')
    print("Record configs finish.")
    print(f'\n **** Run on {args.dataset_file} dataset. **** \n')

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Logger
    local_rank = torch.distributed.get_rank()
    logger = None
    if local_rank == 0:
        long_id = args.exp_name
        logger = TensorboardLogger(long_id, long_id, local_rank)  # id name + time tag
        logger.log_string('hyperpara', str(args))

    model, criterion, postprocessor = build_model(args)

    if args.eval:
        dict = torch.load(args.resume_path, map_location="cpu")
        model.load_state_dict(dict["model"])
    else:
        peft_config = LoraConfig(task_type='OTHER', inference_mode=False, r=16, lora_alpha=32,
                                 lora_dropout=0.1,
                                 target_modules='.*unet.*to_[q,v].*')
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        for n, p in model.named_parameters():
            if not match_name_keywords(n, ['cv_model', 'unet', 'vae', 'text_encoder']):
                p.requires_grad = True
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp=model

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_text_encoder_names)
                 and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,  # 1e-4
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr,  # 1e-4 ['backbone.0']
        },
        # {
        #     "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
        #     "lr": args.lr_backbone,
        # },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,  # 1e-4*1 ['reference_points','sampling_offsets']
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)
    grad_scaler = amp.GradScaler('cuda', enabled=args.amp)
    
    if not args.eval and args.resume:
        model_without_ddp.load_state_dict(dict["model"])
        optimizer.load_state_dict(dict["optimizer"])
        lr_scheduler.load_state_dict(dict["lr_scheduler"])
        grad_scaler.load_state_dict(dict["grad_scaler"])
    print("\n **** Using AMP? {}. **** \n".format(args.amp))

    # train==true or dataset not a2d/jhmdb
    if not (args.eval and (args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb')):
        dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
        if args.distributed:
            if args.cache_mode:
                sampler_train = samplers.NodeDistributedSampler(dataset_train)
            else:
                sampler_train = samplers.DistributedSampler(dataset_train)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # A2D-Sentences/jhmdb
    if args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
        dataset_val = build_dataset(args.dataset_file, image_set='val', args=args)
        if args.distributed:
            if args.cache_mode:
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print("\n **** Missing Keys: {}. **** \n".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("\n **** Unexpected Keys: {}. **** \n".format(unexpected_keys))

    # evaluation of a2d or jhmdb
    if args.eval:
        assert args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb', \
                    'Only A2D-Sentences and JHMDB-Sentences datasets support evaluation'
        print("\n **** Begin to evaluating {}. **** \n".format(args.dataset_file))
        with torch.no_grad():
            test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
        return

    print("\n **** Start training, total poch is: {}, begin from epoch: {}. **** \n".format(args.epochs, args.start_epoch))
    start_time = time.time()
    total_itr_num = 0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 0 and not (args.eval and (args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb')):
            # ****************** Reload dataset ******************
            args.current_epoch = epoch
            dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
            if args.distributed:
                if args.cache_mode:
                    sampler_train = samplers.NodeDistributedSampler(dataset_train)
                else:
                    sampler_train = samplers.DistributedSampler(dataset_train)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)
            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                           collate_fn=utils.collate_fn, num_workers=args.num_workers)
            print("Reload dataset.")

        if args.distributed:
            sampler_train.set_epoch(epoch)

        epoch_s_ = time.time()
        train_stats, total_itr_num = train_one_epoch(
            args, model, criterion, data_loader_train, optimizer, grad_scaler, device, epoch,
            args.clip_max_norm, total_itr_num, lr_scheduler, logger)
        epoch_e_ = time.time()
        print("\n **** Train one epoch time cost is {}h. **** \n".format((epoch_e_-epoch_s_)/3600))

        lr_scheduler.step()

        checkpoint_paths = []
        # ===== 最后一个 epoch 保存合并权重 =====
        if epoch + 1 == args.epochs:
            # merge LoRA 权重
            merged_model = model_without_ddp.merge_and_unload()  # 合并 LoRA 权重
            merged_path = output_dir / "model.pth"
            utils.save_on_master({
                'model': merged_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'grad_scaler': grad_scaler.state_dict(),
                }, merged_path)
            print("Saved merged model for inference")
        
        elif args.output_dir:
            if (epoch + 1) % 1 == 0:
                checkpoint_path=output_dir / f'checkpoint_lora{epoch:04}.pth'
                # ===== 中途保存：未合并权重 =====
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),  # LoRA 未合并
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'grad_scaler': grad_scaler.state_dict(),
                }, checkpoint_path)
                print(f"Saved checkpoint_lora at epoch {epoch}")
                # 只保留最近两个
                checkpoint_paths.append(checkpoint_path)
                if len(checkpoint_paths) > 2:
                    old_checkpoint = checkpoint_paths.pop(0)
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                        print(f"Deleted old checkpoint_lora: {old_checkpoint}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # if args.dataset_file == 'a2d':
        #     print("Begin to evaluating {}...".format(args.dataset_file))
        #     with torch.no_grad():
        #         test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
        #     log_stats.update({**{f'{k}': v for k, v in test_stats.items()}})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\n **** Total training time for this task is {}. **** \n'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HCD training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    gpu_num = torch.cuda.device_count()
    print("Use GPU number is: ", gpu_num)
    if not args.resume:
        args.lr *= gpu_num / 4
        args.lr_backbone *= gpu_num / 4
        args.lr_text_encoder *= gpu_num / 4
    else:
        args.lr *= gpu_num / 8
        args.lr_backbone *= gpu_num / 8
        args.lr_text_encoder *= gpu_num / 8
    print("\n **** After adjust with GPU&BATCH num {}/{}, lr: {}, lr_backbone: {}, lr_text_backbone: {}. **** \n".format(gpu_num, args.batch_size, args.lr, args.lr_backbone, args.lr_text_encoder))
    main(args)



