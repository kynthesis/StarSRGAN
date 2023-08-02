# StarSRGAN: Improving Real-World Blind Super-Resolution
Official PyTorch Implementation of [StarSRGAN: Improving Real-World Blind Super-Resolution](https://arxiv.org/abs/2307.16169)

Accepted for Oral Presentation at the International Conference in Central Europe on 

Computer Graphics, Visualization and Computer Vision 2023 (WSCG 2023).

May 15-19, 2023, Prague/Pilsen, Czech Republic


**Khoa D. Vo, Len T. Bui.**

Faculty of Information Technology (FIT), University of Science, VNU.HCM, Ho Chi Minh City, Vietnam.

# How to Train/Finetune StarSRGAN models

### Overview

The training has been divided into two stages. These two stages have the same data synthesis process and training pipeline, except for the loss functions. Specifically,

1. We first train StarSRNet with L1 loss from the pre-trained model ESRGAN.
1. We then use the trained StarSRNet model as an initialization of the generator, and train the StarSRGAN with a combination of L1 loss, perceptual loss and GAN loss.

### Dataset Preparation

We use DF2K (DIV2K and Flickr2K) for our training. Only HR images are required. <br>
You can download from :

1. DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

Here are steps for data preparation.

#### Step 1: [Optional] Generate multi-scale images

For the DF2K dataset, we use a multi-scale strategy, *i.e.*, we downsample HR images to obtain several Ground-Truth images with different scales. <br>
You can use the [scripts/generate_multiscale_DF2K.py](scripts/generate_multiscale_DF2K.py) script to generate multi-scale images. <br>
Note that this step can be omitted if you just want to have a fast try.

```bash
python scripts/generate_multiscale_DF2K.py --input datasets/DF2K/HR --output datasets/DF2K/multiscale
```

#### Step 2: [Optional] Crop to sub-images

We then crop DF2K images into sub-images for faster IO and processing.<br>
This step is optional if your IO is enough or your disk space is limited.

You can use the [scripts/extract_subimages.py](scripts/extract_subimages.py) script. Here is the example:

```bash
 python scripts/extract_subimages.py --input datasets/DF2K/multiscale --output datasets/DF2K/multiscale_sub --crop_size 400 --step 200
```

#### Step 3: Prepare a txt for meta information

You need to prepare a txt file containing the image paths. The following are some examples in `meta_info.txt` (As different users may have different sub-images partitions, this file is not suitable for your purpose and you need to prepare your own txt file):

```txt
sub/000001_s001.png
sub/000001_s002.png
sub/000001_s003.png
...
```

You can use the [scripts/generate_meta_info.py](scripts/generate_meta_info.py) script to generate the txt file. <br>
You can merge several folders into one meta_info txt. Here is the example:

```bash
 python scripts/generate_meta_info.py --input datasets/DF2K/HR, datasets/DF2K/multiscale --root datasets/DF2K, datasets/DF2K --meta_info datasets/DF2K/meta_info/meta_info.txt
```

### Train StarSRNet

1. Modify the content in the option file `options/train_realesrnet_x4plus.yml` accordingly:
    ```yml
    train:
        name: DF2K
        type: StarSRGANDataset
        dataroot_gt: datasets/DF2K # modify to the root path of your folder
        meta_info: datasets/DF2K/meta_info.txt # modify to your own generate meta info txt
        io_backend:
            type: disk
    ```
1. Before the formal training, you may run in the `--debug` mode to see whether everything is OK. We use four GPUs for training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 starsrgan/train.py -opt options/train_starsrnet.yml --launcher pytorch --debug
    ```

    Train with **a single GPU** in the *debug* mode:
    ```bash
    python starsrgan/train.py -opt options/train_starsrnet.yml --debug
    ```
1. The formal training. We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 starsrgan/train.py -opt options/train_starsrnet.yml --launcher pytorch --auto_resume
    ```

    Train with **a single GPU**:
    ```bash
    python starsrgan/train.py -opt options/train_starsrnet.yml --auto_resume
    ```

### Train Real-ESRGAN

1. After the training of Real-ESRNet, you now have the file `experiments/train_StarSRNet_2M/model/net_g_2000000.pth`. If you need to specify the pre-trained path to other files, modify the `pretrain_network_g` value in the option file `train_starsrgan.yml`.
1. Modify the option file `train_starsrgan.yml` accordingly. Most modifications are similar to those listed above.
1. Before the formal training, you may run in the `--debug` mode to see whether everything is OK. We use four GPUs for training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 starsrgan/train.py -opt options/train_starsrgan.yml --launcher pytorch --debug
    ```

    Train with **a single GPU** in the *debug* mode:
    ```bash
    python starsrgan/train.py -opt options/train_starsrgan_x4plus.yml --debug
    ```
1. The formal training. We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 starsrgan/train.py -opt options/train_starsrgan.yml --launcher pytorch --auto_resume
    ```

    Train with **a single GPU**:
    ```bash
    python starsrgan/train.py -opt options/train_starsrgan.yml --auto_resume
    ```

## Finetune StarESRGAN on your own dataset

You can finetune StarESRGAN on your own dataset. Typically, the fine-tuning process can be divided into two cases:

1. [Generate degraded images on the fly](#Generate-degraded-images-on-the-fly)
1. [Use your own **paired** data](#Use-paired-training-data)

### Generate degraded images on the fly

Only high-resolution images are required. The low-quality images are generated with the degradation process described in Real-ESRGAN during trainig.

**1. Prepare dataset**

See [this section](#dataset-preparation) for more details.

**2. Download pre-trained models**

Download pre-trained models into `experiments/pretrained_models`.

**3. Finetune**

Modify [options/finetune_starsrgan.yml](options/finetune_starsrgan.yml) accordingly, especially the `datasets` part:

```yml
train:
    name: DF2K
    type: StarSRGANDataset
    dataroot_gt: datasets/DF2K # modify to the root path of your folder
    meta_info: datasets/DF2K/meta_info.txt # modify to your own generate meta info txt
    io_backend:
        type: disk
```

We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 starsrgan/train.py -opt options/finetune_starsrgan.yml --launcher pytorch --auto_resume
```

Finetune with **a single GPU**:
```bash
python realesrgan/train.py -opt options/finetune_starsrgan.yml --auto_resume
```
