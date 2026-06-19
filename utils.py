# Code based on the following repositories:
# https://github.com/soupeeli/NeuroBOLT

import os 
import glob
import json
import time
import datetime
from collections import defaultdict, deque
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])

def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])

def setup_for_distributed(is_master):
    """
    Only the master process prints by default.
    Any process can still print if you call print(..., force=True)
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env() 
        args.gpu = _get_local_rank_env()
        args.dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def create_ds_config(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer: 
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }
        writer.write(json.dumps(ds_config, indent=2))

class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        
        self.deque = deque(maxlen=window_size)  
        self.total = 0.0                       
        self.count = 0                         
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Sync count and total across all GPUs. 
        Warning: deque is NOT synchronized
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        # Auto-creates a SmoothedValue for any new metric
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            
            if isinstance(v, torch.Tensor):
                v = v.item()
            
            assert isinstance(v, (float, int)), \
                f"Metric values must be scalar, got {type(v)}"
            
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, optimizer_disc=None):
    output_dir = args.output_dir

    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, f'checkpoint-{latest_ckpt}.pth')
            print(f"Auto resume checkpoint: {args.resume}")

        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print(f"Resume checkpoint {args.resume}")
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                args.start_epoch = 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, f'checkpoint-{latest_ckpt}')
                print(f"Auto resume checkpoint: {latest_ckpt}")
                _, client_states = model.load_checkpoint(args.output_dir, tag=f'checkpoint-{latest_ckpt}')
                args.start_epoch = client_states['epoch'] + 1

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, optimizer_disc=None):
    epoch_name = str(epoch)
    subdir_name = f"{args.dataset}-{args.model}"
    output_dir = os.path.join(args.output_dir, subdir_name)
    os.makedirs(output_dir, exist_ok=True)

    if not getattr(args, 'enable_deepspeed', False):
        checkpoint_path = os.path.join(output_dir, f'checkpoint-{epoch_name}.pth')
    
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        if loss_scaler is not None:
            to_save['scaler'] = loss_scaler.state_dict()

        if optimizer_disc is not None:
            to_save['optimizer_disc'] = optimizer_disc.state_dict()

        save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag=f"checkpoint-{epoch_name}", client_state=client_state)

class MultiObjectiveLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        B, P = pred.shape
        
        # Point-wise MSE Loss
        L_MSE = F.mse_loss(pred, target)

        # Temporal Correlation Loss (per ROI)
        pred_t   = pred   - pred.mean(dim=0, keepdim=True)   
        target_t = target - target.mean(dim=0, keepdim=True)
        num_t    = (pred_t * target_t).sum(dim=0)             
        den_t    = (pred_t.pow(2).sum(dim=0) *
                    target_t.pow(2).sum(dim=0)).sqrt()
        R_t      = num_t / (den_t + 1e-8)               
        L_TCORR   = (1 - R_t.clamp(-1, 1)).mean()

        # Spatial Correlation Loss (per timepoint)
        pred_s   = pred   - pred.mean(dim=1, keepdim=True)   
        target_s = target - target.mean(dim=1, keepdim=True) 
        num_s    = (pred_s * target_s).sum(dim=1)       
        den_s    = (pred_s.pow(2).sum(dim=1) *          
                    target_s.pow(2).sum(dim=1)).sqrt()
        R_s      = num_s / (den_s + 1e-8)                   
        L_SCORR  = (1 - R_s.clamp(-1, 1)).mean() 

        L_MO = self.alpha * L_MSE + self.beta * L_TCORR + (1 - self.alpha - self.beta) * L_SCORR
            
        loss_dict = {
            'loss_total': L_MO.item(),
            'loss_mse': L_MSE.item(),
            'loss_tcorr': L_TCORR.item(),
            'loss_scorr': L_SCORR.item(),
        }
        
        return L_MO, loss_dict

def corr_metric(x, y):
    assert x.shape == y.shape, f'{x.shape} and {y.shape}'
    r = np.corrcoef(x.squeeze(), y.squeeze())[0, 1]
    return r

def mse_metric(x, y):
    assert x.shape == y.shape, f'{x.shape} and {y.shape}'
    return mean_squared_error(x, y)

def fc_corr(x):
    x_c = x - x.mean(axis=0, keepdims=True)
    cov = x_c.T @ x_c / (x.shape[0] - 1)
    std = np.sqrt(np.diag(cov) + 1e-8)
    corr = cov / (std[:, None] * std[None, :] + 1e-8)
    mask = np.triu(np.ones((corr.shape[0], corr.shape[0]), dtype=bool), k=1)
    return corr[mask]

def fc_mse_metric(x, y):
    return mse_metric(fc_corr(x), fc_corr(y))

def pixcorr_metric(x, y):
    return corr_metric(fc_corr(x), fc_corr(y))

def edge_f1_metric(x, y, threshold_percentile=75):
    corr_pred = fc_corr(x)
    corr_true = fc_corr(y)
    threshold = np.percentile(np.abs(corr_true), threshold_percentile)
    true_binary = (np.abs(corr_true) >= threshold).astype(int)
    pred_binary = (np.abs(corr_pred) >= threshold).astype(int)
    return f1_score(true_binary, pred_binary)