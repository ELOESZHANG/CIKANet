import numpy as np
import cv2
import os
from scipy.io import loadmat

satellite = 'wv3'
if satellite == 'wv3':
    global BGR
    BGR = [1,2,4]
elif satellite == 'qb':
    BGR = [0,1,2]
elif satellite == 'wv2':
    BGR = [1,2,4]
RR = f"./fused/{satellite.upper()}/RR/"
FR = f"./fused/{satellite.upper()}/FR/"


def show16(windowname, img, bitnum=11):
    # img = np.uint16(img)
    if len(img.shape) == 3:
        img = img.transpose(1, 2, 0)
        w, h, bands = img.shape
    else:
        w, h = img.shape
        bands = 1
    min, max = np.zeros(bands), np.zeros(bands)
    for i in range(bands):
        hist = cv2.calcHist([img], [i], None, histSize=[2**bitnum], ranges=[0, 2**bitnum-1])
        p = 0
        while p < 0.02 and min[i] <= 2**bitnum-1:
            p += hist[int(min[i])]/(w*h)
            min[i] = min[i]+1
        p = 0
        while p < 0.98 and max[i] <= 2**bitnum-1:
            p += hist[int(max[i])]/(w*h)
            max[i] = max[i]+1
    img = np.clip((img.astype(np.float32)-min)/(max-min), 0, 1)
    cv2.imshow(windowname, img)


def load_image_r(filepath):
    data = loadmat(filepath)
    return {
        'gt': data['gt'],
        'ms': data['ms'],
        'pan': data['pan'],
        'sr': data['sr']
    }


def load_image_f(filepath):
    data = loadmat(filepath)
    return {
        'ms': data['ms'],
        'pan': data['pan'],
        'sr': data['sr']
    }


length = len(os.listdir(RR))
for i in range(length):
    dataset_r = load_image_r(RR+f"{i+1}.mat")
    sr = dataset_r['sr'][BGR,:,:]
    gt = dataset_r['gt'][BGR,:,:]
    pan = dataset_r['pan']
    ms = dataset_r['ms'][BGR,:,:]
    show16("sr", sr)
    show16("ms", ms)
    show16("gt", gt)
    show16("pan", pan)
    cv2.waitKey()
cv2.destroyAllWindows()
length = len(os.listdir(FR))
for i in range(length):
    dataset_f = load_image_f(FR+f"{i+1}.mat")
    sr = dataset_f['sr'][BGR,:,:]
    pan = dataset_f['pan']
    ms = dataset_f['ms'][BGR,:,:]
    show16("sr", sr)
    show16("ms", ms)
    show16("pan", pan)
    cv2.waitKey()