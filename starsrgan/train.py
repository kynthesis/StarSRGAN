# flake8: noqa
import os.path as osp
from starsrgan.utils.train import train_pipeline

import starsrgan.archs
import starsrgan.data
import starsrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
