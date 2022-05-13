from utils.common import *
import numpy as np
import torch
import os

class dataset:
    def __init__(self, dataset_dir, subset):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.data = torch.Tensor([])
        self.labels = torch.Tensor([])
        self.data_file = os.path.join(self.dataset_dir, f"data_{self.subset}.npy")
        self.labels_file = os.path.join(self.dataset_dir, f"labels_{self.subset}.npy")
        self.cur_idx = 0
    
    def generate(self, lr_crop_size, hr_crop_size, transform=False):      
        if exists(self.data_file) and exists(self.labels_file):
            print(f"{self.data_file} and {self.labels_file} HAVE ALREADY EXISTED\n")
            return
        data = []
        labels = []
        step = hr_crop_size - 1

        subset_dir = os.path.join(self.dataset_dir, self.subset)
        ls_images = sorted_list(subset_dir)
        for image_path in ls_images:
            print(image_path)
            hr_image = read_image(image_path)

            h = hr_image.shape[1]
            w = hr_image.shape[2]
            for x in np.arange(start=0, stop=h-hr_crop_size, step=step):
                for y in np.arange(start=0, stop=w-hr_crop_size, step=step):
                    subim_label  = hr_image[:, x : x + hr_crop_size, y : y + hr_crop_size]
                    if transform:
                        subim_label = random_transform(subim_label)
 
                    subim_data = gaussian_blur(subim_label, sigma=0.7)
                    subim_data = resize_bicubic(subim_data, lr_crop_size, lr_crop_size)

                    subim_label = rgb2ycbcr(subim_label)
                    subim_data = rgb2ycbcr(subim_data)

                    subim_label = norm01(subim_label)
                    subim_data = norm01(subim_data)
   
                    data.append(subim_data.numpy())
                    labels.append(subim_label.numpy())

        data = np.array(data)
        labels = np.array(labels)
        
        np.save(self.data_file, data)
        np.save(self.labels_file, labels)

    def load_data(self):
        if not exists(self.data_file):
            raise ValueError(f"\n{self.data_file} and {self.labels_file} DO NOT EXIST\n")
        self.data = np.load(self.data_file)
        self.data = torch.as_tensor(self.data)
        self.labels = np.load(self.labels_file)
        self.labels = torch.as_tensor(self.labels)
    
    def get_batch(self, batch_size, shuffle_each_epoch=True):
        # Ignore remaining dataset because of  
        # shape error when run torch.mean()
        isEnd = False
        if self.cur_idx + batch_size > self.data.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                self.data, self.labels = shuffle(self.data, self.labels)
        
        data = self.data[self.cur_idx : self.cur_idx + batch_size]
        labels = self.labels[self.cur_idx : self.cur_idx + batch_size]
        self.cur_idx += batch_size
        
        return data, labels, isEnd
