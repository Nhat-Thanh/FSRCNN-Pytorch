from utils.common import *
from model import FSRCNN 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',      type=int, default=2,                   help='-')
parser.add_argument("--ckpt-path",  type=str, default="",                  help='-')
parser.add_argument("--image-path", type=str, default="dataset/test1.png", help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path
ckpt_path = FLAGS.ckpt_path
scale = FLAGS.scale

if scale not in [2, 3, 4]:
    raise ValueError("must be 2, 3 or 4")

if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/x{scale}/FSRCNN-x{scale}.pt"

sigma = 0.3 if scale == 2 else 0.2


# -----------------------------------------------------------
# demo
# -----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr_image = read_image(image_path)
    bicubic_image = upscale(lr_image, scale)
    write_image("bicubic.png", bicubic_image)

    lr_image = gaussian_blur(lr_image, sigma=sigma)
    lr_image = rgb2ycbcr(lr_image)
    lr_image = norm01(lr_image)
    lr_image = torch.unsqueeze(lr_image, dim=0)

    model = FSRCNN(scale, device)
    model.load_weights(ckpt_path)
    with torch.no_grad():
        lr_image = lr_image.to(device)
        sr_image = model.predict(lr_image)[0]

    sr_image = denorm01(sr_image)
    sr_image = sr_image.type(torch.uint8)
    sr_image = ycbcr2rgb(sr_image)

    write_image("sr.png", sr_image)

if __name__ == "__main__":
    main()
