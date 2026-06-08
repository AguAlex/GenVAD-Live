from collections.abc import Iterable
import numpy as np
import torch
from sklearn import metrics
from util import misc
from util.abnormal_utils import filt

def inference(model: torch.nn.Module, data_loader: Iterable,
                   device: torch.device,
                   log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing Inference'

    predictions = []
    labels = []
    videos = []
    
    for data_iter_step, (samples, targets, label, vid, _) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())
        
        samples = samples.to(device)
        targets = targets.to(device)
        
        # Rulăm modelul și cerem anomalia
        _, _, _, recon_error = model(samples, mask_ratio=args.mask_ratio)
        recon_error = recon_error.detach().cpu().numpy()
        predictions += list(recon_error)

    predictions = np.array(predictions)
    labels = np.array(labels)
    videos = np.array(videos)

    evaluate_model(predictions, labels, videos, range=38, mu=11, normalize_scores=False)

def evaluate_model(predictions, labels, videos,
                   range=38, mu=11, normalize_scores=False):

    aucs = []
    filtered_preds = []
    filtered_labels = []
    
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = filt(pred, range=range, mu=mu)
        
        if normalize_scores:
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-6)

        pred = np.nan_to_num(pred, nan=0.)

        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)
        
        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)

    macro_auc = np.nanmean(aucs)

    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    print(f"Final MicroAUC: {micro_auc:.4f}, Final MacroAUC: {macro_auc:.4f}")
    return micro_auc, macro_auc