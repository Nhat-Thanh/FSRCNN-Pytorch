from neuralnet import FSRCNN_model
from utils.common import exists
import torch
import numpy as np

# -----------------------------------------------------------
#  SRCNN
# -----------------------------------------------------------

class FSRCNN:
    def __init__(self, scale, device):
        self.device = device
        self.model = FSRCNN_model(scale).to(device)
        self.optimizer = None
        self.loss =  None
        self.metric = None
        self.model_path = None
        self.ckpt_path = None
        self.ckpt_man = None

    def setup(self, optimizer, loss, metric, model_path, ckpt_path):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        # @the best model weights
        self.model_path = model_path
        self.ckpt_path = ckpt_path

    def load_checkpoint(self, ckpt_path):
        if not exists(ckpt_path):
            return
        self.ckpt_man = torch.load(ckpt_path)
        self.optimizer.load_state_dict(self.ckpt_man['optimizer'])
        self.model.load_state_dict(self.ckpt_man['model'])

    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=torch.device(self.device)))

    def predict(self, lr):
        self.model.train(False)
        sr = self.model(lr)
        return sr

    def evaluate(self, dataset, batch_size=64):
        losses, metrics = [], []
        isEnd = False
        while isEnd == False:
            lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr = self.predict(lr)
            loss = self.loss(hr, sr).cpu()
            metric = self.metric(hr, sr).cpu()
            losses.append(loss.detach().numpy())
            metrics.append(metric.detach().numpy())

        metric = np.mean(metrics)
        loss = np.mean(losses)
        return loss, metric

    def train(self, train_set, valid_set, batch_size, 
              steps, save_every=1, save_best_only=False):

        cur_step = 0
        if self.ckpt_man is not None:
            cur_step = self.ckpt_man['step']
        max_steps = cur_step + steps

        prev_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_path)

        loss_mean = []
        metric_mean = []
        while cur_step < max_steps:
            cur_step += 1
            lr, hr, _ = train_set.get_batch(batch_size)
            loss, metric = self.train_step(lr, hr)
            loss_mean.append(loss.detach().numpy())
            metric_mean.append(metric.detach().numpy())

            if (cur_step % save_every == 0) or (cur_step >= max_steps):
                val_loss, val_metric = self.evaluate(valid_set)
                print(f"Step {cur_step}/{max_steps}",
                      f"- loss: {np.mean(loss_mean):.7f}",
                      f"- {self.metric.__name__}: {np.mean(metric_mean):.3f}",
                      f"- val_loss: {val_loss:.7f}",
                      f"- val_{self.metric.__name__}: {val_metric:.3f}")
                loss_mean = []
                metric_mean = []
                torch.save({'step': cur_step,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()
                            }, self.ckpt_path)

                if save_best_only and val_loss > prev_loss:
                    continue
                prev_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Save model to {self.model_path}\n")
  
    def train_step(self, lr, hr):
        self.model.train(True)
        self.optimizer.zero_grad()

        lr, hr = lr.to(self.device), hr.to(self.device)
        sr = self.model(lr)

        loss = self.loss(hr, sr)
        loss.backward()

        self.optimizer.step()

        metric = self.metric(hr, sr)
        loss = loss.cpu()
        metric = metric.cpu()
        return loss, metric
