from data_feeder import ImageLoader, ImageCollate
from torch.utils.data import DataLoader
from models import FCN
import os
import Config
from torch.nn import functional as F
import torch
from torch.autograd import Variable
from tqdm import tqdm

GPU = torch.cuda.is_available()
def calc_acc(prediction, target):
    num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
    acc = 100.0 * num_corrects/Config.batch_size
    return acc

def validate(model, test_loader):
    acc = 0
    with torch.no_grad():
        for i, (batch, target) in enumerate(test_loader):
            prediction = model(batch)
            acc += calc_acc(prediction.cpu().detach(), target.cpu())
        acc = acc.numpy()/(i+1)
    return acc


infilestrain = [os.path.join(path, fi) for path, subdirs, files in os.walk(Config.Train_path) for fi in files if fi.endswith(Config.ext)]
infilestest = [os.path.join(path, fi) for path, subdirs, files in os.walk(Config.Test_path) for fi in files if fi.endswith(Config.ext)]
classes = {}
i = 0
for img_path in infilestrain:
    if img_path.split('/')[1] not in classes.keys():
        classes[img_path.split('/')[-2]] = i
        i += 1

collate_fn = ImageCollate(Config.input_size, classes, Config.input_chan, GPU)
trainset = ImageLoader(infilestrain, Config)
testset = ImageLoader(infilestest, Config)
train_loader = DataLoader(trainset, num_workers=0, shuffle=Config.shuffle, sampler=None, batch_size=Config.batch_size, pin_memory=False, drop_last=True, collate_fn=collate_fn)
test_loader = DataLoader(testset, num_workers=0, shuffle=Config.shuffle, sampler=None, batch_size=Config.batch_size, pin_memory=False, drop_last=True, collate_fn=collate_fn)

loss_fn = F.cross_entropy
best_acc = 0
model = FCN(Config, len(list(classes.keys())))
if GPU:
    model.cuda()
model.train()
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.learning_rate, weight_decay=1e-6)
for epoch in range(Config.epochs):
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            optim.zero_grad()
            model.zero_grad()
            prediction = model(batch)
            loss = loss_fn(prediction, target)
            acc = calc_acc(prediction.cpu().detach(), target.cpu())
            l1_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for x in model.parameters():
                l1_reg = l1_reg + (Config.reg * x.data.norm(1))
            l1_reg.backward()
            loss.backward()
            optim.step()
            tepoch.set_postfix(loss=loss.item(), accuracy= acc)
    model.eval()
    result = validate(model, test_loader)
    if result > best_acc:
        best_acc = result
        save_path = os.path.join(Config.checkpoint_path, "epoch_{}".format(epoch))
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}, save_path)
    print(f"Best Accuracy is {best_acc} in {epoch}th epoch")