import os
import time
import datetime
import argparse
import pickle
import json
import numpy as np 

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler, SequentialSampler
from torch.nn.parallel import DistributedDataParallel
from timm.models import create_model

import utils
import optim
import models.marble
from engine import train_epoch, evaluate

def get_args():
    parser = argparse.ArgumentParser('Marble Training Script', add_help=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='min lr for cyclic schedulers that hit 0 (default: 1e-6)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of LR warmup steps, will overload warmup_epochs if set > 0')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPT',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPS',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None)

    parser.add_argument('--model', default='marble', type=str, metavar='MODEL')
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--n_channels', default=26, type=int,
                        help='Number of EEG channels when training from scratch. If fine-tuning a model with '
                             'pre-trained weights, ensure the channel number matches the original model.')
    parser.add_argument('--n_roi', default=64, type=int)

    parser.add_argument('--train_test_mode', default='full_test', type=str,
                        help='full_test/full_retainvu/')
    parser.add_argument('--dataset', default='VU', type=str)
    parser.add_argument('--prepro_datapath', default='../../data/neurobolt/preprocessed/vu.pkl', type=str)
    parser.add_argument('--output_dir', default='./checkpoints/')
    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=False)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    known_args, _ = parser.parse_known_args()
    if known_args.enable_deepspeed:
        import deepspeed 
        parser = deepspeed.add_config_arguments(parser)
        ds_init = deepspeed.initialize
    else:
        ds_init = None
    
    return parser.parse_args(), ds_init

def get_dataset(args, verbose=True):
    ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
                'P7', 'P8', 'FPZ', 'FZ', 'CZ', 'PZ', 'POZ', 'OZ', 'FT9', 'FT10', 'TP9', 'TP10']
    try: 
        with open(args.prepro_datapath, 'rb') as file: 
            if args.train_test_mode == 'full_retainvu':
                splits = pickle.load(file)
            else: 
                train_data, val_data, test_data, _ = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {args.prepro_datapath}")
    except Exception as e:
        raise RuntimeError(f"Error loading data from {args.prepro_datapath}: {e}")
    
    if args.train_test_mode == 'full_retainvu':
        eeg_train  = torch.cat([s[0] for s in splits["train"]], dim=0)
        fmri_train = torch.cat([s[1] for s in splits["train"]], dim=0)
        train_dataset = TensorDataset(eeg_train, fmri_train)
        val_dataset  = splits["val"]   
        test_dataset = splits["test"]
        subset_ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
                            'P7', 'P8', 'FZ', 'CZ', 'PZ', 'OZ', 'TP9', 'TP10', 'POZ']
        subset_ch_ids = [ch_names.index(ch) for ch in subset_ch_names]

        train_samples = len(train_dataset)
        val_samples  = sum(s[0].shape[0] for s in val_dataset)
        test_samples = sum(s[0].shape[0] for s in test_dataset)

    else:
        eeg_train_tensor, fmri_train_tensor = train_data
        eeg_val_tensor, fmri_val_tensor = val_data
        eeg_test_tensor, fmri_test_tensor = test_data

        train_dataset = TensorDataset(eeg_train_tensor, fmri_train_tensor)
        val_dataset = TensorDataset(eeg_val_tensor, fmri_val_tensor)
        test_dataset = TensorDataset(eeg_test_tensor, fmri_test_tensor)

        train_samples = len(train_dataset)
        val_samples = len(val_dataset)
        test_samples = len(test_dataset)

        subset_ch_ids = None

    # Optional: Log dataset creation
    if verbose: 
        print("Datasets successfully created:")
        print(f"  Train: {train_samples} samples")
        print(f"  Validation: {val_samples} samples")
        print(f"  Test: {test_samples} samples")

    return train_dataset, val_dataset, test_dataset, ch_names, subset_ch_ids

def get_model(args):
    if args.model == "marble":
        model = create_model(
            args.model,
            emb_size=args.emb_size,
            n_roi=args.n_roi,
            depth=args.depth,
            n_heads=args.n_heads,
            n_channels=args.n_channels,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model

def collate_variable_length(batch):
    eegs, fmris = zip(*batch)
    return eegs[0], fmris[0]

def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)
    
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_dataset, val_dataset, test_dataset, ch_names, subset_ch_ids = get_dataset(args, verbose=True)
    
    global_rank = utils.get_rank()
    num_tasks = utils.get_world_size()
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print(f"Train sampler = {train_sampler}")
    if len(val_dataset) % num_tasks != 0:
        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
    
    if args.dataset == "VU" and args.train_test_mode == 'full_test':
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)

        bs_val = len(val_dataset) // 5
        bs_test = len(test_dataset) // 6

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, 
                                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=bs_val, 
                                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=bs_test, 
                                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    
    elif args.train_test_mode == 'full_retainvu':
        val_sampler  = SequentialSampler(range(len(val_dataset)))
        test_sampler = SequentialSampler(range(len(test_dataset)))
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, 
                                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=args.num_workers, 
                                    pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_variable_length)
        test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=args.num_workers, 
                                    pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_variable_length)
    
    model = get_model(args).to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters: {n_parameters}")

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    niter_per_epoch = len(train_dataset) // total_batch_size
    print(f"LR = {args.lr:.6f}")
    print(f"Batch size = {total_batch_size}")
    print(f"Update frequent = {args.update_freq}")
    print(f"Number of training examples = {len(train_dataset)}") 
    print(f"Number of training training per epoch = {niter_per_epoch}")
    
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = optim.get_parameter_groups(model, args.weight_decay)
        model, optimizer, _, _ = ds_init(args=args, model=model, model_parameters=optimizer_params, 
                                         dist_init_required=not args.distributed)
        print(f"model.gradient_accumulation_steps() = {model.gradient_accumulation_steps()}")
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed: 
            model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        
        optimizer = optim.create_optimizer(args, model_without_ddp)
        
        loss_scaler = optim.NativeScalerWithGradNormCount()
    
    lr_schedule_values = optim.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, niter_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    wd_schedule_values = optim.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, niter_per_epoch
    )

    criterion = utils.MultiObjectiveLoss()
    print(f"Criterion = {str(criterion)}")

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler
    )

    if args.eval:
        test_stats = evaluate(test_dataloader, model, device, header='Test:', 
                              subset_ch_ids=subset_ch_ids, is_binary=(args.n_roi == 1))
        print(f"Test stats:")
        for key in ["mse", "corr", "fc_mse", "pixcorr", "f1"]:
            print(f"{key}: {test_stats[key]:.4f} ± {test_stats[key+'_std']:.4f}")
        exit(0)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = float('inf')
    max_test_corr = 0.0
    max_test_pixcorr = 0.0
    for epoch in range(args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        start_steps = epoch * niter_per_epoch
        train_stats = train_epoch(train_dataloader, model, criterion, optimizer, device, epoch, loss_scaler, 
                                  args.clip_grad, start_steps, lr_schedule_values, wd_schedule_values, 
                                  niter_per_epoch, args.update_freq, (args.n_roi == 1), subset_ch_ids)
        val_stats = evaluate(val_dataloader, model, device, header='Val:', 
                              subset_ch_ids=subset_ch_ids, is_binary=(args.n_roi == 1))
        print("Validation stats:")
        for key in ["mse", "corr", "fc_mse", "pixcorr", "f1"]:
            print(f"{key}: {val_stats[key]:.4f} ± {val_stats[key+'_std']:.4f}")
        test_stats = evaluate(test_dataloader, model, device, header='Test:', 
                              subset_ch_ids=subset_ch_ids, is_binary=(args.n_roi == 1))
        print("Test stats:")
        for key in ["mse", "corr", "fc_mse", "pixcorr", "f1"]:
            print(f"{key}: {test_stats[key]:.4f} ± {test_stats[key+'_std']:.4f}")
        
        epoch_name = f"{epoch}-mse{test_stats['mse']:.4f}-corr{test_stats['corr']:.4f}-pixcorr{test_stats['pixcorr']:.4f}-f1{test_stats['f1']:.4f}"
        if val_stats['loss'] < min_loss:
            min_loss = val_stats['loss']
            max_test_corr = test_stats['corr']
            max_test_pixcorr = test_stats['pixcorr']
            if args.output_dir and test_stats['corr'] > 0:
                print(f"New best model found at epoch {epoch}, saving model...")
                utils.save_model(args, epoch_name, model, model_without_ddp, optimizer, loss_scaler)
        
        print(f'Min val loss: {min_loss:.2f}, Max test corr: {max_test_corr:.2f}, Max test pixcorr: {max_test_pixcorr:.2f}')
        log_stats = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                     **{f'val_{k}': float(v) for k, v in val_stats.items()},
                     **{f'test_{k}': float(v) for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
        
if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        os.makedirs(opts.output_dir, exist_ok=True)
    main(opts, ds_init)

