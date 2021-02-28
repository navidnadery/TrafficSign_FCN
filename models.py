import torch
import torch.nn as nn
from torch.autograd import Variable

class FCN(nn.Module):
    def __init__(self, params, out_size):
        super().__init__()
        layers = []
        input_size = params.input_size
        input_chan = params.input_chan
        self.feature_size = params.cnn_dim[-1]
        for i, (cnn_filter, n_filters) in enumerate(zip(params.cnn_kernel, params.cnn_dim)):
            layers += [(nn.Conv2d(input_chan, n_filters, cnn_filter, stride=1, padding=0))]
            layers += [nn.BatchNorm2d(n_filters), nn.ReLU()]
            layers += [nn.Dropout(p=params.dr)]
            input_chan = n_filters
            input_size += 1 - cnn_filter
            if i<len(params.cnn_pool) and params.cnn_pool[i] > 1:
                layers += [nn.MaxPool2d(kernel_size=params.cnn_pool[i],stride=params.cnn_pool[i])]
                input_size /= params.cnn_pool[i]

        layers += [nn.AvgPool2d(kernel_size=int(input_size),stride=int(input_size))]
        self.FixedCNNs = nn.Sequential(*layers)
        ## if you want to handle the sequence or layers inner values uncomment bellow and comment the previous line (FixedCNNs)
        # self.CNNs = nn.ModuleList(layers)
        self.Classifier = nn.Linear(params.cnn_dim[-1], out_size)
        self.Label = nn.Softmax()

    def forward(self, x):
        hid = self.FixedCNNs(x)
        ## if you want to handle the sequence or layers inner values uncomment bellow and comment the previous line (FixedCNNs)
        # for cnn in self.CNNs:
        #     x = cnn(x)
        output = self.Classifier(hid.view(-1,self.feature_size))
        label = self.Label(output)
        return label