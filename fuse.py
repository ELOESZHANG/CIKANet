from model import *
import numpy as np
from osgeo import gdal
import torch
from scipy.io import savemat
import os
import warnings
import tqdm
warnings.filterwarnings("ignore")

device_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
satellite = 'WV3'
max_value = 2047.
if satellite == 'WV3':
    global ch
    ch = 8
elif satellite == 'qb':
    ch = 4
elif satellite == 'wv2':
    ch = 8
method = 'cikanet'
FR_test_dir = f'./data/{satellite.upper()}/test/Full Resolution/'
FR_save_dir = f"./fused/{satellite}/FR/"
RR_test_dir = f'./data/{satellite.upper()}/test/Reduced Resolution/'
RR_save_dir = f"./fused/{satellite}/RR/"


if not os.path.exists(FR_save_dir):
    os.makedirs(FR_save_dir)
if not os.path.exists(RR_save_dir):
    os.makedirs(RR_save_dir)

    
def load_image(img_path):
    dataset = gdal.Open(img_path)
    img = dataset.ReadAsArray()
    img = img.astype(np.float32)
    return img


def read_data_r(root, index):
    ms = load_image(root+f'MS/{index}')
    pan = load_image(root + f'PAN/{index}')
    gt = load_image(root + f'GT/{index}')
    return {
        'gt':  gt,
        'ms':  ms,
        'pan': pan
    }


def read_data_f(root, index):
    ms = load_image(root + f'MS/{index}')
    pan = load_image(root + f'PAN/{index}')
    return {
        'ms': ms,
        'pan': pan
    }


def forward(model, pan, ms):
    ms = torch.from_numpy(ms).unsqueeze(0).type(torch.float32)
    pan = torch.from_numpy(pan).unsqueeze(0).unsqueeze(0).type(torch.float32)

    ms = ms.cuda()
    pan = pan.cuda()
    result = model(ms, pan)
    result = result.cpu()
    result = result.numpy()[0]
    return result


index = os.listdir(FR_test_dir+'MS/')
with torch.no_grad():
    model = CIKANet(ch, 256).cuda()
    model.load_state_dict(torch.load("./model/pretrain_model.pth".format(method, satellite))['model'])
    model.eval()
    for i in tqdm.tqdm(range(len(index))):
        dataset_r = read_data_r(RR_test_dir, str(i+1)+'.TIF')
        pan = dataset_r['pan']/max_value
        ms = dataset_r['ms']/max_value
        result = forward(model, pan, ms)
        result = np.clip(result, 0, 1) * max_value
        savemat(RR_save_dir + f"{i+1}.mat", {'sr': result, 'gt': dataset_r['gt'], 'pan': dataset_r['pan'], 'ms': dataset_r['ms']})

    model = CIKANet(ch, 256).cuda()
    model.load_state_dict(torch.load("./model/pretrain_model.pth".format(method, satellite))['model'])
    model.eval()
    for i in tqdm.tqdm(range(len(index))):
        dataset_f = read_data_f(FR_test_dir, str(i+1)+'.TIF')
        pan = dataset_f['pan']/max_value
        ms = dataset_f['ms']/max_value
        H, W = pan.shape
        C, *_ = ms.shape
        result = np.zeros((C, H, W), dtype=np.float32)
        for j in range(4):
            for k in range(4):
                result[:, j*256:(j+1)*256, k*256:(k+1)*256] = forward(model, pan[j*256:(j+1)*256, k*256:(k+1)*256], ms[:, j*64:(j+1)*64, k*64:(k+1)*64])
        result = np.clip(result, 0, 1) * max_value
        savemat(FR_save_dir + f"{i + 1}.mat", {'sr': result, 'pan': dataset_f['pan'], 'ms': dataset_f['ms']})
