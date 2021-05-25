import math
from collections import OrderedDict
import torch
import torch.nn.functional as F

class SPPLayer(torch.nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super().__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        batch_size, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size_h = math.ceil(h / 2 ** i)
            kernel_size_w = math.ceil(w / 2 ** i)
            kernel_size = (kernel_size_h, kernel_size_w)
            padding_h = math.floor((kernel_size_h * 2 ** i - h + 1)/2)
            padding_w = math.floor((kernel_size_w * 2 ** i - w + 1)/2)
            padding = (padding_h, padding_w)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size, padding=padding).view(batch_size, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size, padding=padding).view(batch_size, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class ImgNetSPP(torch.nn.Module):
    def __init__(self, channel=3, spp_level=3):
        super().__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2 ** (i * 2)
        print(self.num_grids)

        self.conv1 = torch.nn.Conv2d(channel, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # padding=1保证无论如何也能取到整个像素集，从而不用在forward中进行判断
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        self.relu = torch.nn.ReLU()

        self.spp_layer = SPPLayer(spp_level)

        self.linear_model = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(self.num_grids * 32, 64)),
            ('fc1_relu', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(64, 50)),
        ]))

    def forward(self, x):
        x = self.relu((self.pooling(self.conv1(x))))
        x = self.relu((self.pooling(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.spp_layer(x)
        x = self.linear_model(x)
        return x


if __name__ == '__main__':
    import sys
    sys.path.append('/home/qlwang/img_cap')

    from util.imgCapDataset import *

    img_cap = ImgCapDataset()
    img_cap_loader = DataLoader(img_cap, batch_size=1, shuffle=True)
    model = ImgNetSPP()

    for i, img_caption in enumerate(img_cap_loader, 0):
        # prepare img&label
        img, caption = img_caption
        y_50_pred = model(img)
        print(y_50_pred)

        