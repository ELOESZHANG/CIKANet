# CIKANet
This project is the code for "Cross-Interaction Kernel Attention Network For Pansharpening"
## Test and Visualize
run `python fuse.py` to use pretrained model for pansharpening

run `python imgshow.py` to visualize the fused images
## How To Train?
**Step1. Put datasets**
* Put datasets (WorldView3, QuickBird, WorldView2) into the `/data`, see following path structure. 
```
|-$ROOT/data
├── WV3
│   ├── train
│   │   ├── MS
│   │   │   ├── 1.TIF
│   │   ├── ├──...
│   │   ├── PAN
│   │   │   ├── 1.TIF
│   │   ├── ├──...
│   │   ├── GT
│   │   │   ├── 1.TIF
│   │   ├── ├──...
│   ├── valid
│   │   ├── MS
│   │   │   ├── 1.TIF
│   │   ├── ├──...
│   │   ├── PAN
│   │   │   ├── 1.TIF
│   │   ├── ├──...
│   │   ├── GT
│   │   │   ├── 1.TIF
│   │   ├── ├──...
│   ├── test
│   │   ├── Reduced Resolution
│   │   │   ├── MS
│   │   │   │   ├── 1.TIF
│   │   │   ├── ├──...
│   │   │   ├── PAN
│   │   │   │   ├── 1.TIF
│   │   │   ├── ├──...
│   │   │   ├── GT
│   │   │   │   ├── 1.TIF
│   │   │   ├── ├──...
│   │   ├── Reduced Resolution
│   │   │   ├── MS
│   │   │   │   ├── 1.TIF
│   │   │   ├── ├──...
│   │   │   ├── PAN
│   │   │   │   ├── 1.TIF
│   │   │   ├── ├──...
```
All the MS and GT images are with the shape of (C, H/r, W/r).
All the PAN images are with the shape of (H, W).

**Step2. train**

run `python train.py` for training
