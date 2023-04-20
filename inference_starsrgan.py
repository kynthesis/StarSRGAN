import os
import cv2
import glob
import argparse


from starsrgan.utils.enhancer import Enhancer
from starsrgan.archs.generator_arch import RealSRNet, StarSRNet, LiteSRNet


def main():
    """Inference demo for StarSRGAN."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="inputs", help="Input image or folder"
    )
    parser.add_argument("-n", "--model_name", type=str)
    parser.add_argument(
        "-o", "--output", type=str, default="results", help="Output folder"
    )
    parser.add_argument(
        "-s",
        "--outscale",
        type=float,
        default=4,
        help="The final upsampling scale of the image",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Option] Model path. Usually, you do not need to specify it",
    )
    parser.add_argument(
        "--suffix", type=str, default="out", help="Suffix of the restored image"
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=0,
        help="Tile size, 0 for no tile during testing",
    )
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding")
    parser.add_argument(
        "--pre_pad", type=int, default=0, help="Pre padding size at each border"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 precision during inference. Default: fp16 (half precision)",
    )
    parser.add_argument(
        "--alpha_upsampler",
        type=str,
        default="realesrgan",
        help="The upsampler for the alpha channels. Options: realesrgan | bicubic",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="auto",
        help="Image extension. Options: auto | jpg | png, auto means using the same extension as inputs",
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        default=None,
        help="GPU device to use (default=None) can be 0,1,2 for multi-gpu",
    )
    parser.add_argument(
        "-z",
        "--drop-out",
        type=bool,
        default=False,
        help="Use drop-out degradation. Default: False",
    )

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split(".")[0]
    if args.model_name.lower().startswith('real'):
        model = RealSRNet()
    elif args.model_name.lower().startswith('lite'):
        model = LiteSRNet(drop_out=args.drop_out)
    else:
        model = StarSRNet(drop_out=args.drop_out)
    netscale = 4

    # determine model paths
    model_path = os.path.join(
        "experiments", args.model_name, "models", "net_g_latest.pth"
    )

    # restorer
    upsampler = Enhancer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id,
    )

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, "*")))

    # measure inference time
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float16).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = len(paths)
    timings = np.zeros((repetitions, 1))

    # warm-up GPU
    for _ in range(100):
        _ = model(dummy_input)

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print("Testing", idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        else:
            img_mode = None

        try:
            starter.record()
            output, _ = upsampler.enhance(img, outscale=args.outscale)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[idx] = curr_time
        except RuntimeError as error:
            print("Error", error)
            print(
                "If you encounter CUDA out of memory, try to set --tile with a smaller number."
            )
        else:
            if args.ext == "auto":
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == "RGBA":  # RGBA images should be saved in png format
                extension = "png"
            if args.suffix == "":
                save_path = os.path.join(args.output, f"{imgname}.{extension}")
            else:
                save_path = os.path.join(
                    args.output, f"{imgname}_{args.suffix}.{extension}"
                )
            cv2.imwrite(save_path, output)

    mean_syn = np.sum(timings) / repetitions
    print(str(round(1000 / mean_syn, 2)) + " FPS")


if __name__ == "__main__":
    main()
