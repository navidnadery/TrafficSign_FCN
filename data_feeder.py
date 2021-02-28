import random
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

class ImageLoader(data.Dataset):
    def __init__(self, image_paths, params):
        self.image_paths = image_paths
        self.batch_size = params.batch_size
        self.image_size = params.input_size
        random.seed(params.seed)
        random.shuffle(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image_reshaped = np.array(image.resize([self.image_size,self.image_size]))
        return (image_reshaped, self.image_paths[index].split('/')[-2])

    def __len__(self):
        return len(self.image_paths)

class ImageCollate():
    def __init__(self, input_size, classes, nchannels, GPU):
        self.input_size = input_size
        self.CLASS = classes
        self.channels = nchannels
        self.GPU = GPU

    def __call__(self, batch):
        raw_data = torch.FloatTensor(len(batch), self.channels, self.input_size, self.input_size)
        raw_data.zero_()
        target = torch.LongTensor(len(batch))
        for i, item in enumerate(batch):
            raw_data[i, :, :, :] = torch.from_numpy(item[0].transpose(2,0,1))
            target[i] = self.CLASS[item[1]]

        return ((raw_data.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).long()) if self.GPU else (raw_data, target))