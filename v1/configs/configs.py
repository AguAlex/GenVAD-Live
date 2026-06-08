import argparse
import os

def get_configs_avenue():
    parser = argparse.ArgumentParser(description="GenVAD Avenue Configs")
    
    # 1. Dataset & Paths (Rute)
    parser.add_argument('--dataset', type=str, default='avenue')
    parser.add_argument('--avenue_path', type=str, default='./Avenue_Dataset')
    parser.add_argument(
        '--avenue_gt_path',
        type=str,
        default=None,
        help="implicit: <avenue_path>/ground_truth — folosești alt path pentru UCSD (ex. TXT ground truth)",
    )
    parser.add_argument('--output_dir', type=str, default='./Checkpoints')
    
    # 2. Training Hyperparameters (Hiperparametri Antrenare)
    parser.add_argument('--run_type', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=8) # Redus de la 100 pt Colab (evităm Out of Memory)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--start_TS_epoch', type=int, default=50)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--print_freq', type=int, default=10)
    
    # 3. Model Architecture & Masking (Arhitectura)
    parser.add_argument('--model', type=str, default='mae_cvt')
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--masking_method', type=str, default='random_masking')
    parser.add_argument('--norm_pix_loss', type=bool, default=False)
    parser.add_argument('--use_only_masked_tokens_ab', type=bool, default=False)
    parser.add_argument('--grad_weighted_rec_loss', type=bool, default=True)
    parser.add_argument('--input_3d', type=bool, default=True)
    
    # 4. Hardware & Loaders (Placa video și procesor)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4) # Redus de la 10 pt Colab
    parser.add_argument('--pin_mem', type=bool, default=False)
    
    # 5. Anomalii Sintetice
    # SETEAZĂ LA 0.25 DOAR DUPĂ CE POPULEZI FOLDERELE frames_abnormal / masks_abnormal
    parser.add_argument('--percent_abnormal', type=float, default=0.0) 
    
    args, _ = parser.parse_known_args()

    if args.avenue_gt_path is None:
        args.avenue_gt_path = os.path.join(args.avenue_path, "ground_truth")
    
    # 6. Parametri compuși (Liste și Tupluri)
    args.abnormal_score_func = ['L2', 'L2']
    args.input_size = (320, 640) # Rezoluția originală folosită pentru Avenue
    
    return args