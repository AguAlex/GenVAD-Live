import argparse
import os


def get_configs():
    parser = argparse.ArgumentParser(
        description="GenVAD v4_convlstm — Future Frame + ConvLSTM (UCSD / Avenue)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ucsd",
        choices=("ucsd", "avenue"),
    )
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument(
        "--ucsd_path",
        type=str,
        default=os.environ.get("UCSD_PATH", "/content/UCSD_Dataset"),
    )
    parser.add_argument(
        "--avenue_path",
        type=str,
        default=os.environ.get("AVENUE_PATH", "/content/Avenue_Dataset"),
    )
    parser.add_argument("--avenue_gt_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./v4_convlstm_outputs")

    parser.add_argument(
        "--run_type",
        type=str,
        default="train_eval",
        choices=("train_eval", "train", "eval"),
    )
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_input_frames", type=int, default=4)
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Dimensiune hidden ConvLSTM.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Număr straturi ConvLSTM stacked.",
    )
    parser.add_argument(
        "--encoder_channels",
        type=int,
        default=32,
        help="Canale după encoder per cadru.",
    )

    parser.add_argument(
        "--resize",
        type=str,
        default="",
        help="WxH. Gol = auto (UCSD 384x256, Avenue 640x320).",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print_freq", type=int, default=50)

    parser.add_argument("--filt_range", type=int, default=-1)
    parser.add_argument("--filt_mu", type=int, default=-1)
    parser.add_argument("--no_filt", action="store_true")
    parser.add_argument("--eval_each_epoch", action="store_true")

    args, _ = parser.parse_known_args()

    if not args.data_path:
        args.data_path = (
            args.ucsd_path if args.dataset == "ucsd" else args.avenue_path
        )
    if not args.avenue_gt_path:
        args.avenue_gt_path = os.path.join(args.data_path, "ground_truth")

    if not args.resize:
        args.resize = "384x256" if args.dataset == "ucsd" else "640x320"

    if args.filt_range < 0 or args.filt_mu < 0:
        if args.dataset == "avenue":
            args.filt_range = 38 if args.filt_range < 0 else args.filt_range
            args.filt_mu = 11 if args.filt_mu < 0 else args.filt_mu
        else:
            args.filt_range = 302 if args.filt_range < 0 else args.filt_range
            args.filt_mu = 21 if args.filt_mu < 0 else args.filt_mu

    return args
