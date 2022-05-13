import os
from neuralnet import FSRCNN_model 
from utils.common import exists, tensor2numpy
import torch
import numpy as np

class logger:
    def __init__(self, path, values) -> None:
        self.path = path
        self.values = values

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
            loss = self.loss(hr, sr)
            metric = self.metric(hr, sr)
            losses.append(tensor2numpy(loss))
            metrics.append(tensor2numpy(metric))

        metric = np.mean(metrics)
        loss = np.mean(losses)
        return loss, metric

    def train(self, train_set, valid_set, batch_size, steps, save_every=1,
              save_best_only=False, save_log=False, log_dir=None):

        if (save_log) and (log_dir is None):
            raise ValueError("log_dir must be specified if save_log is True")
        os.makedirs(log_dir, exist_ok=True)
        dict_logger = {"loss":       logger(path=os.path.join(log_dir, "losses.npy"),      values=[]),
                       "metric":     logger(path=os.path.join(log_dir, "metrics.npy"),     values=[]),
                       "val_loss":   logger(path=os.path.join(log_dir, "val_losses.npy"),  values=[]),
                       "val_metric": logger(path=os.path.join(log_dir, "val_metrics.npy"), values=[])}
        for key in dict_logger.keys():
            path = dict_logger[key].path
            if exists(path):
                dict_logger[key].values = np.load(path).tolist()

        cur_step = 0
        if self.ckpt_man is not None:
            cur_step = self.ckpt_man['step']
        max_steps = cur_step + steps

        prev_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_path)

        loss_buffer = []
        metric_buffer = []
        while cur_step < max_steps:
            cur_step += 1
            lr, hr, _ = train_set.get_batch(batch_size)
            loss, metric = self.train_step(lr, hr)
            loss_buffer.append(tensor2numpy(loss))
            metric_buffer.append(tensor2numpy(metric))

            if (cur_step % save_every == 0) or (cur_step >= max_steps):
                loss = np.mean(loss_buffer)
                metric = np.mean(metric_buffer)
                val_loss, val_metric = self.evaluate(valid_set)
                print(f"Step {cur_step}/{max_steps}",
                      f"- loss: {loss:.7f}",
                      f"- {self.metric.__name__}: {metric:.3f}",
                      f"- val_loss: {val_loss:.7f}",
                      f"- val_{self.metric.__name__}: {val_metric:.3f}")
                if save_log == True:
                    dict_logger["loss"].values.append(loss)
                    dict_logger["metric"].values.append(metric)
                    dict_logger["val_loss"].values.append(val_loss)
                    dict_logger["val_metric"].values.append(val_metric)
                
                loss_buffer = []
                metric_buffer = []
                torch.save({'step': cur_step,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()
                            }, self.ckpt_path)

                if save_best_only and val_loss > prev_loss:
                    continue
                prev_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Save model to {self.model_path}\n")
        
        if save_log == True:
            for key in dict_logger.keys():
                logger_obj = dict_logger[key]
                path = logger_obj.path
                values = np.array(logger_obj.values, dtype=np.float32)
                np.save(path, values)
  
    def train_step(self, lr, hr):
        self.model.train(True)
        self.optimizer.zero_grad()

        lr, hr = lr.to(self.device), hr.to(self.device)
        sr = self.model(lr)

        loss = self.loss(hr, sr)
        metric = self.metric(hr, sr)
        loss.backward()
        self.optimizer.step()

        return loss, metric
