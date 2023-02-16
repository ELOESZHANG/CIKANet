# coding=gbk
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import random
from model import *
from osgeo import gdal
import torch
import warnings
import tqdm
warnings.filterwarnings("ignore")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# configure
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 123
set_random_seed(seed)
max_value = 2047.
satellite = 'WV3'
method = 'cikanet'
train_batch_size = 16
valid_batch_size = 16
total_epochs = 800
lr = 1e-3
num_workers = 4

if satellite == 'WV3':
    global ch
    ch = 8
elif satellite == 'qb':
    ch = 4
elif satellite == 'wv2':
    ch = 8

# filepath setting
traindata_dir = f'./data/{satellite.upper()}/train/'
validdata_dir = f'./data/{satellite.upper()}/valid/'
trainrecord_dir = './records/trainloss-{}-{}/'.format(method, satellite)
validrecord_dir = './records/validloss-{}-{}/'.format(method, satellite)
model_dir = './results/models-{}-{}/'.format(method, satellite)
checkpoint_model = model_dir + 'checkpoint.pth'

if not os.path.exists(trainrecord_dir):
    os.makedirs(trainrecord_dir)
if not os.path.exists(validrecord_dir):
    os.makedirs(validrecord_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def load_image(img_path):
    dataset = gdal.Open(img_path)
    img = dataset.ReadAsArray()
    img = img.astype(np.float32) / max_value
    return img


# get dataset
class ReducedDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.image_filenames = [x for x in os.listdir(img_dir + 'MS')]
        random.shuffle(self.image_filenames)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        ms = load_image(self.img_dir + 'MS/{}'.format(self.image_filenames[index]))
        pan = load_image(self.img_dir + 'PAN/{}'.format(self.image_filenames[index]))
        gt = load_image(self.img_dir + 'GT/{}'.format(self.image_filenames[index]))
        if self.transform:
            ms = self.transform(ms)
            gt = self.transform(gt)
            pan = self.transform(pan)
        return ms, pan, gt


# Matrix to Tensor
class ToTensor(object):
    def __call__(self, input):
        if len(input.shape)==3:
            input = torch.from_numpy(input).type(torch.FloatTensor)
        else:
            input = torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor)
        return input


def get_train_set(traindata_dir):
    return ReducedDataset(traindata_dir, transform=transforms.Compose([ToTensor()]))


def get_valid_set(validdata_dir):
    return ReducedDataset(validdata_dir, transform=transforms.Compose([ToTensor()]))


# model training and validating
def train(model, trainset, validset, start_epoch, criterion, optimizer, best_epoch,
          best_valid_avg_loss, scheduler):
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    print('----- Begin Training!')
    trainset_dataloader = DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True,
                                     pin_memory=True, drop_last=True)
    validset_dataloader = DataLoader(dataset=validset, batch_size=valid_batch_size, shuffle=False,
                                     pin_memory=True, drop_last=True)
    train_steps_per_epoch = len(trainset_dataloader)
    valid_steps_per_epoch = len(validset_dataloader)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch + 1, total_epochs + 1):
        # training stage
        train_total_loss = 0

        model.train()
        print(f"target_lr: {optimizer.state_dict()['param_groups'][0]['lr']}")
        for batch, (ms, pan, gt) in tqdm.tqdm(enumerate(trainset_dataloader)):
            ms = ms.cuda()
            pan = pan.cuda()
            gt = gt.cuda()
            with torch.cuda.amp.autocast():
                sr = model(ms, pan)
                train_loss = criterion(sr, gt)

            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total_loss += train_loss.item()

        # validating stage
        valid_total_loss = 0
        model.eval()
        for batch, (ms, pan, gt) in enumerate(validset_dataloader):
            ms = ms.cuda()
            pan = pan.cuda()
            gt = gt.cuda()
            with torch.cuda.amp.autocast():
                sr = model(ms, pan)
                valid_loss = criterion(sr, gt)
            valid_total_loss += valid_loss.item()

        # save model's paramaters
        train_avg_loss = train_total_loss / train_steps_per_epoch
        valid_avg_loss = valid_total_loss / valid_steps_per_epoch
        print('Epoch[{}/{}]: train_loss: {:.15f} valid_loss: {:.15f}'.format(epoch, total_epochs,
                                                                             train_avg_loss, valid_avg_loss))
        with open('%s/train_loss_record.txt' % trainrecord_dir, "a") as train_loss_record:
            train_loss_record.write(
                "Epoch[{}/{}]: train_loss: {:.15f}\n".format(epoch, total_epochs, train_avg_loss))
        with open('%s/valid_loss_record.txt' % validrecord_dir, "a") as valid_loss_record:
            valid_loss_record.write(
                "Epoch[{}/{}]: valid_loss: {:.15f}\n".format(epoch, total_epochs, valid_avg_loss))
        if best_valid_avg_loss > valid_avg_loss:
            torch.save({'model': model.state_dict()}, model_dir + f'best_epoch.pth')
            best_epoch = epoch
            best_valid_avg_loss = valid_avg_loss
            print(f"best epoch = {best_epoch}")
        else:
            print(f"best epoch = {best_epoch}")
        state = {'model': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(),
                 'best_epoch': best_epoch, 'best_valid_avg_loss': best_valid_avg_loss}
        torch.save(state, checkpoint_model)
        torch.save(state, model_dir + f'{epoch}.pth')
        scheduler.step()


def main():
    model = CIKANet(ch).cuda()
    criterion = nn.MSELoss().cuda()

    transformed_trainset = get_train_set(traindata_dir)
    transformed_validset = get_valid_set(validdata_dir)
    print('train:', len(transformed_trainset))
    print('valid:', len(transformed_validset))

    # Device setting
    params = sum(p.numel() for p in list(model.parameters())) / 1e3
    print('#Params: %.1fK' % (params))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(checkpoint_model):
        print("----- loading checkpoint '{}'".format(checkpoint_model))
        checkpoint = torch.load(checkpoint_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_valid_avg_loss = checkpoint['best_valid_avg_loss']
        print('----- load epoch {} successfully!'.format(start_epoch))
    else:
        start_epoch = 0
        best_epoch = 1
        best_valid_avg_loss = np.inf
        print('----- no model exists, training from the begining!')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=400, gamma=0.1, last_epoch=start_epoch-1)
    train(model, transformed_trainset, transformed_validset, start_epoch, criterion, optimizer, best_epoch,
          best_valid_avg_loss, scheduler)


if __name__ == '__main__':
    main()