from utils.dataset import dataset
from utils.common import PSNR
from model import FSRCNN
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=100000,          help='-')
parser.add_argument("--scale",          type=int, default=2,               help='-')
parser.add_argument("--batch-size",     type=int, default=128,             help='-')
parser.add_argument("--save-every",     type=int, default=100,             help='-')
parser.add_argument("--save-best-only", type=int, default=0,               help='-')
parser.add_argument("--save-log",       type=int, default=0,               help='-')
parser.add_argument("--ckpt-dir",       type=str, default="checkpoint/x2", help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAG, unparsed = parser.parse_known_args()
steps = FLAG.steps
batch_size = FLAG.batch_size
save_every = FLAG.save_every
save_log = (FLAG.save_log == 1)
save_best_only = (FLAG.save_best_only == 1)

scale = FLAG.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3 or 4")

ckpt_dir = FLAG.ckpt_dir
if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = f"checkpoint/x{scale}"
model_path = os.path.join(ckpt_dir, f"FSRCNN-x{scale}.pt")
ckpt_path = os.path.join(ckpt_dir, f"ckpt.pt")



# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
lr_crop_size = 10
hr_crop_size = lr_crop_size * scale

train_set = dataset(dataset_dir, "train")
train_set.generate(lr_crop_size, hr_crop_size)
train_set.load_data()

valid_set = dataset(dataset_dir, "validation")
valid_set.generate(lr_crop_size, hr_crop_size)
valid_set.load_data()


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fsrcnn = FSRCNN(scale, device)
    fsrcnn.setup(optimizer=torch.optim.Adam(fsrcnn.model.parameters(), lr=1e-3),
                loss=torch.nn.MSELoss(), model_path=model_path,
                ckpt_path=ckpt_path, metric=PSNR)

    fsrcnn.load_checkpoint(ckpt_path)
    fsrcnn.train(train_set, valid_set, steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, save_every=save_every,
                save_log=save_log, log_dir=ckpt_dir)

if __name__ == "__main__":
    main()
