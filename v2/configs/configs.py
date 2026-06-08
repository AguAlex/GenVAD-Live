import argparse

def get_configs_avenue():
    parser = argparse.ArgumentParser(description="GenVAD Avenue Configs Simplified")
    
    parser.add_argument('--dataset', type=str, default='avenue')
    parser.add_argument('--avenue_path', type=str, default='./Avenue_Dataset')
    parser.add_argument('--output_dir', type=str, default='./Checkpoints')
    
    parser.add_argument('--run_type', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--resume', type=str, default='')
    
    parser.add_argument('--model', type=str, default='mae_cvt')
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--masking_method', type=str, default='random_masking')
    parser.add_argument('--use_only_masked_tokens_ab', type=bool, default=False)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pin_mem', type=bool, default=False)
    
    # Parametru pentru anomaliile tale sintetice
    parser.add_argument('--percent_abnormal', type=float, default=0.25) # default=0.25

    parser.add_argument('--print_freq', type=int, default=10)
    
    args, _ = parser.parse_known_args()
   # args.input_size = (320, 640) 
    args.input_size = (256, 384)
    return args