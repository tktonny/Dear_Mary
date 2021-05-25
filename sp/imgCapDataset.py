import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class ImgCapDataset(Dataset):
    def __init__(self):
        # 图片数据集路径
        self.img_paths= '/home/qlwang/img_cap/res/image'
        self.img_paths_list = os.listdir(self.img_paths)
        print(self.img_paths_list)

        # caption路径
        self.caption_path = '/home/qlwang/img_cap/res/caption/caption.txt'

        # 图片
        self.imgs = []
        
        # totensor=>转为tensor后归一化
        self.tran1 = transforms.ToTensor()

        # 加载图片数据集
        for img_name in self.img_paths_list:
            img_path = os.path.join(self.img_paths, img_name)
            img = cv2.imread(img_path)
            img_tensor = self.tran1(img)
            self.imgs.append(img_tensor)
            print(img_path)
        
        # 加载caption
        caption = np.loadtxt(self.caption_path, delimiter='\t', dtype=str)
        caption = pd.DataFrame(caption, columns=['cap', 'name'])
        print(caption)
        self.captions = [caption.loc[caption['name'] == name, 'cap'].values[0] for name in self.img_paths_list]
        print(self.captions)
        del caption

        # 每个通道计算图片数据集的均值、方差
        pix_rearrange = []
        for img_tensor in self.imgs:
            pixes = img_tensor.view(3, -1)
            pix_rearrange.append(pixes)
        pix_rearrange = torch.cat(pix_rearrange, dim = 1)
        mean = pix_rearrange.mean(dim=1)
        std = pix_rearrange.std(dim=1)
        print(mean, std)
        print(self.imgs[0])
        del pix_rearrange

        # normalize => 中心化+标准化
        self.tran2 = transforms.Normalize(mean, std)
        for i in range(len(self.imgs)):
            self.imgs[i] = self.tran2(self.imgs[i])

        self.len = len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index], self.captions[index]
    
    def __len__(self):
        return self.len

if __name__ == '__main__':
    img_cap = ImgCapDataset()
    img_cap_loader = DataLoader(img_cap, batch_size=1, shuffle=True)

    for i, img_caption in enumerate(img_cap_loader, 0):
        # prepare img&label
        img, caption = img_caption
        print(img, caption)
