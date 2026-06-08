import argparse
import datetime
import json
import os
import time
from pathlib import Path

from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter

from configs.configs import get_configs_avenue
# NOTĂ: Asigură-te că dataset-urile tale nu mai returnează 'gradients'
from data.train_dataset import AbnormalDatasetTrain
from data.test_dataset import AbnormalDatasetTest
from engine_train import train_one_epoch, test_one_epoch
from inference import inference
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8
from util import misc
import torch

def main(args):
    print("Configs: \n{}".format(args))
    log_writer = SummaryWriter(log_dir=args.output_dir)
    device = args.device

    if args.run_type == 'train':
        dataset_train = AbnormalDatasetTrain(args)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
        )

    dataset_test = AbnormalDatasetTest(args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    if args.dataset == 'avenue':
        model = mae_cvt_patch16(img_size=args.input_size, 
                                use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                masking_method=args.masking_method).float()
    else:
        model = mae_cvt_patch8(img_size=args.input_size,
                               use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                               masking_method=args.masking_method).float()
        
    model.to(device)

    if args.run_type == "train":
        do_training(args, data_loader_test, data_loader_train, device, log_writer, model)
    elif args.run_type == "inference":
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint-best.pth"))
        model.load_state_dict(checkpoint['model'], strict=False)
        with torch.no_grad():
            inference(model, data_loader_test, device, args=args)

def do_training(args, data_loader_test, data_loader_train, device, log_writer, model):
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
    
    best_micro = 0.0
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)
        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        test_stats = test_one_epoch(model, data_loader_test, device, epoch, log_writer=log_writer, args=args)
        log_stats_test = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}

        misc.save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, latest=True)
        
        if test_stats.get('micro', 0) > best_micro:
            best_micro = test_stats['micro']
            misc.save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, best=True)

        if log_writer:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a") as f:
                f.write(json.dumps(log_stats_train) + "\n" + json.dumps(log_stats_test) + "\n")

    print('Training time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avenue') 
    args, _ = parser.parse_known_args()

    args = get_configs_avenue()
    if args.dataset == 'custom':
        args.custom_path = "./Custom_Dataset" 
        
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)