import argparse
import os


def get_configs():
    parser = argparse.ArgumentParser(
        description="GenVAD v3 — MOG2 (UCSD Ped2 / CUHK Avenue)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ucsd",
        choices=("ucsd", "avenue"),
        help="ucsd = Ped2 (.tif + _gt bmp); avenue = .png + .mat/.txt GT.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Rădăcina dataset-ului. Dacă e gol, se folosește --ucsd_path sau --avenue_path.",
    )
    parser.add_argument(
        "--ucsd_path",
        type=str,
        default=os.environ.get("UCSD_PATH", "/content/UCSD_Dataset"),
        help="Rădăcină UCSD (când --dataset ucsd și --data_path gol).",
    )
    parser.add_argument(
        "--avenue_path",
        type=str,
        default=os.environ.get("AVENUE_PATH", "/content/Avenue_Dataset"),
        help="Rădăcină Avenue (când --dataset avenue și --data_path gol).",
    )
    parser.add_argument(
        "--avenue_gt_path",
        type=str,
        default="",
        help="Folder ground_truth .txt (gol = <data_path>/ground_truth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./v3_outputs",
        help="Rezultate, log-uri, opțional măști debug.",
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="eval",
        choices=("eval", "fit_only"),
        help="eval = antrenează MOG2 pe train + scor pe test; fit_only = doar fundal.",
    )

    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--var_threshold", type=float, default=16.0)
    parser.add_argument(
        "--detect_shadows",
        action="store_true",
        help="Umbrele MOG2 (127) nu intră în scor.",
    )
    parser.add_argument("--fg_threshold", type=int, default=200)
    parser.add_argument("--morph_kernel", type=int, default=3)

    parser.add_argument(
        "--score_mode",
        type=str,
        default="fg_ratio",
        choices=("fg_ratio", "fg_pixels", "max_blob_area"),
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="",
        help='Redimensionare "WxH". Avenue: ex. "640x320" (ca v2).',
    )
    parser.add_argument("--train_warmup_frames", type=int, default=0)

    parser.add_argument(
        "--filt_range",
        type=int,
        default=-1,
        help="Filtru temporal; -1 = automat (Avenue 38, UCSD 302).",
    )
    parser.add_argument(
        "--filt_mu",
        type=int,
        default=-1,
        help="Filtru temporal mu; -1 = automat (Avenue 11, UCSD 21).",
    )
    parser.add_argument("--no_filt", action="store_true")
    parser.add_argument("--save_debug_masks", action="store_true")
    parser.add_argument("--debug_every", type=int, default=30)

    args, _ = parser.parse_known_args()

    if not args.data_path:
        args.data_path = (
            args.ucsd_path if args.dataset == "ucsd" else args.avenue_path
        )
    if not args.avenue_gt_path:
        args.avenue_gt_path = os.path.join(args.data_path, "ground_truth")

    if args.filt_range < 0 or args.filt_mu < 0:
        if args.dataset == "avenue":
            if args.filt_range < 0:
                args.filt_range = 38
            if args.filt_mu < 0:
                args.filt_mu = 11
        else:
            if args.filt_range < 0:
                args.filt_range = 302
            if args.filt_mu < 0:
                args.filt_mu = 21

    return args
