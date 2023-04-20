import importlib
from os import path as osp
from starsrgan.utils.misc import scandir

# automatically scan and import loss dataset for registry
# scan all the files under the 'data' folder and collect files ending with '_dataset.py'
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(data_folder)
    if v.endswith("_dataset.py")
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f"starsrgan.data.{file_name}")
    for file_name in dataset_filenames
]
