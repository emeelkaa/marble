# Code based on the following repositories:
# https://github.com/soupeeli/NeuroBOLT

import sys
import math
import torch
import numpy as np
import utils

def train_epoch(dataloader, model, criterion, optimizer, device, epoch, loss_scaler, 
                max_norm, start_steps, lr_schedule_values, wd_schedule_values, 
                niter_per_epoch, update_freq, is_binary=False, subset_ch_ids=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    model.train()
 
    if loss_scaler is None: 
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    
    for iter_step, batch in enumerate(metric_logger.log_every(dataloader, 50, 'Epoch: [{}]'.format(epoch))):
        step = iter_step // update_freq
        if step >= niter_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        if (lr_schedule_values is not None or wd_schedule_values is not None) and iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        inputs = batch[0].float().to(device, non_blocking=True) / 100
        targets = batch[-1].float().to(device, non_blocking=True)
        if is_binary:
            targets = targets.unsqueeze(-1)
        
        if loss_scaler is None:
            inputs = inputs.half()
            outputs, attn = model(inputs, input_chans=subset_ch_ids)
            loss, loss_dict = criterion(outputs, targets)
        else: 
            with torch.amp.autocast('cuda', enabled=False):
                outputs, attn = model(inputs, input_chans=subset_ch_ids)
                loss, loss_dict = criterion(outputs, targets)
    
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            grad_norm = None
            loss_scale_value = model.optimizer.loss_scale if hasattr(model.optimizer, "loss_scale") else model.optimizer.cur_scale
        else:
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), 
                                    update_grad=(iter_step + 1) % update_freq == 0)
            if (iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]
        
        torch.cuda.synchronize()

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        if is_binary:
            corr_scores = np.mean(utils.corr_metric(outputs, targets))
        else:
            corr_scores = np.mean([utils.corr_metric(outputs[:, roi], targets[:, roi]) 
                                   for roi in range(outputs.shape[-1])])
        
        metric_logger.update(loss_mse=loss_dict['loss_mse'])
        metric_logger.update(loss_tcorr=loss_dict['loss_tcorr'])
        metric_logger.update(loss_scorr=loss_dict['loss_scorr'])
        metric_logger.update(corr=corr_scores)
        #metric_logger.update(loss_scale=loss_scale_value)

        lr_value = 0.
        for group in optimizer.param_groups:
            lr_value = max(lr_value, group["lr"])
        metric_logger.update(lr=lr_value)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(dataloader, model, device, header='Test:', subset_ch_ids=None, 
             is_binary=False, save_npy=False):
    criterion = utils.MultiObjectiveLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()

    all_pred = []
    all_gt = []
    all_attn = []

    for step, batch in enumerate(metric_logger.log_every(dataloader, 10, header)):
        inputs = batch[0].float().to(device, non_blocking=True) / 100
        targets = batch[-1].float().to(device, non_blocking=True)

        if is_binary:
            targets = targets.unsqueeze(-1)
        
        with torch.amp.autocast('cuda', enabled=False): 
            outputs, attn = model(inputs, input_chans=subset_ch_ids)
            loss, loss_dict = criterion(outputs, targets)
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        attn = attn.cpu().numpy()

        all_pred.append(outputs)
        all_gt.append(targets)
        all_attn.append(attn)

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_mse=loss_dict['loss_mse'])
        metric_logger.update(loss_tcorr=loss_dict['loss_tcorr'])
        metric_logger.update(loss_scorr=loss_dict['loss_scorr'])
        
    metric_logger.synchronize_between_processes()
    
    all_corr = []
    all_fc_mse = []
    all_pixcorr = []
    all_f1 = []
    
    for sub_idx in range(len(all_pred)):
        pred = all_pred[sub_idx]
        gt = all_gt[sub_idx]
        print(f"Subject {sub_idx}: pred shape = {pred.shape}")
        corr_scores = [utils.corr_metric(pred[:, roi], gt[:, roi]) 
                       for roi in range(pred.shape[-1])]
        all_corr.append(np.mean(corr_scores))
        all_fc_mse.append(utils.fc_mse_metric(pred, gt))
        all_pixcorr.append(utils.pixcorr_metric(pred, gt))
        all_f1.append(utils.edge_f1_metric(pred, gt))
    
    all_pred_concat = np.concatenate(all_pred, axis=0)
    all_gt_concat = np.concatenate(all_gt, axis=0)
    all_attn_concat = np.concatenate(all_attn, axis=0)
    mse_scores = [utils.mse_metric(all_pred_concat[:, roi], all_gt_concat[:, roi]) 
                  for roi in range(all_pred_concat.shape[-1])]
    if save_npy: 
        np.save('pred.npy', all_pred_concat)
        np.save('gt.npy', all_gt_concat)
        np.save('attn.npy', all_attn_concat)
    
    return {
        "mse":      float(np.mean(mse_scores)),
        "mse_std":  float(np.std(mse_scores)),
        "corr":     float(np.mean(all_corr)),
        "corr_std": float(np.std(all_corr)),
        "fc_mse":   float(np.mean(all_fc_mse)),
        "fc_mse_std": float(np.std(all_fc_mse)),
        "pixcorr":  float(np.mean(all_pixcorr)),
        "pixcorr_std": float(np.std(all_pixcorr)),
        "f1":    float(np.mean(all_f1)),
        "f1_std": float(np.std(all_f1)),
        "loss": metric_logger.loss.global_avg,
    }
    
