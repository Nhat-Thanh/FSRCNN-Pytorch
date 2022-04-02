from utils.common import *
from model import FSRCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int,   default=2,                            help='-')
parser.add_argument("--image-path",   type=str,   default="dataset/test2.png",          help='-')
parser.add_argument("--ckpt-path",    type=str,   default="checkpoint/x2/FSRCNN-x2.pt", help='-')

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path
ckpt_path = FLAGS.ckpt_path
scale = FLAGS.scale
device = 'cpu'

if scale not in [2, 3, 4]:
    ValueError("must be 2, 3 or 4")


# -----------------------------------------------------------
#  read image and save bicubic image
# -----------------------------------------------------------

lr_image = read_image(image_path)
bicubic_image = upscale(lr_image, scale)
write_image("bicubic.png", bicubic_image)


# -----------------------------------------------------------
# preprocess lr image 
# -----------------------------------------------------------

lr_image = gaussian_blur(lr_image, sigma=0.3)
lr_image = rgb2ycbcr(lr_image)
lr_image = norm01(lr_image)
lr_image = torch.unsqueeze(lr_image, dim=0)


# -----------------------------------------------------------
#  predict and save image
# -----------------------------------------------------------

model = FSRCNN(scale, device)
model.load_weights(ckpt_path)
sr_image = model.predict(lr_image)[0]

sr_image = denorm01(sr_image)
sr_image = sr_image.type(torch.uint8)
sr_image = ycbcr2rgb(sr_image)

write_image("sr.png", sr_image)
