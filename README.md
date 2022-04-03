# [Pytorch] Faster SRCNN

Implementation of FSRCNN model in **Accelerating the Super-Resolution Convolutional Neural Network** paper with Pytorch.

Tensorflow version: https://github.com/Nhat-Thanh/FSRCNN-TF

I used Adam with optimize tuned hyperparameters instead of SGD + Momentum. 

I implemented 3 models, FSRCNN-x2, FSRCNN-x3, FSRCNN-x4.


## Contents
- [Train](#train)
- [Demo](#demo)
- [References](#references)


## Train
You run this command to begin the training:
```
python train.py  --steps=300000             \
                 --scale=2                  \
                 --batch_size=128           \
                 --save-best-only=0         \
                 --save-every=1000          \
                 --ckpt-dir="checkpoint/x2" 
```
- **save-best-only**: if it is **0**, model weights will be saved every **save-every** steps.


**NOTE**: if you want to re-train a new model, you should delete all files in sub-directories in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for the next times, so you can train a model on Google Colab without taking care of GPU time limit.

I trained 3 models on Google Colab in 300000 steps:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15Wlw-avTJWGJ69hIDOkuGoeY86IyVVpl?usp=sharing)

You can get the models here:
- [FSRCNN-x2.pt](checkpoint/x2/FSRCNN-x2.pt)
- [FSRCNN-x3.pt](checkpoint/x3/FSRCNN-x3.pt)
- [FSRCNN-x4.pt](checkpoint/x4/FSRCNN-x4.pt) 



## Demo 
After Training, you can test models with this command, the result is the **sr.png**.
```
python demo.py --image-path="dataset/test2.png"          \
               --ckpt-path="checkpoint/x4/FSRCNN-x4.h5"  \
               --scale=4
```

I evaluated models with Set5, Set14, BSD100 and Urban100 dataset by PSNR. I use Set5's Butterfly to show my result:

<div align="center">

|   Model   |   Set5  |  Set14  | BSD100  | Urban100 |
|:---------:|:-------:|:-------:|:-------:|:--------:|
| FSRCNN-x2 | 37.7191 | 34.0454 | 33.9893 |	31.3276  |
| FSRCNN-x3 | 34.6114 |	31.2628 | 31.3051 | X |
| FSRCNN-x4 | 31.8877 | 29.2617 | 29.5976 |	26.9266  |

  <br/>

  <img src="./README/example.png" width="1000"/><br/>
  <b>Bicubic x2-x3-x4 (top), FSRCNN x2-x3-x4 (bottom).</b>
</div>

## References
- Accelerating the Super-Resolution Convolutional Neural Network: https://arxiv.org/abs/1608.00367
- T91, General100: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
- Set5: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set5_SR.zip
- Set14: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set14_SR.zip
- BSD100: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/BSD100_SR.zip
- Urban100: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Urban100_SR.zip
