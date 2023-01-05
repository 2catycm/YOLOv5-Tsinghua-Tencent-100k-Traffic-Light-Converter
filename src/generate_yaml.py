# 输入
# 数据集
# 输出
# 转换：1. xyxy->xywh 2. label name -> label id 3. 生成dataset.yaml 4. 生成 train.txt val.txt test.txt

# %%
import imagesize
from joblib import memory
import time
from datetime import datetime
# from sklearn.utils import Bunch
from munch import DefaultMunch
import numpy as np
import tomli
from pathlib import Path

import torch
this_file = Path(__file__).absolute()
this_directory = this_file.parent
# _current_time = f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
_current_time = f"_{datetime.now().strftime('%Y-%m-%d-%H')}"

# %%
memory = memory.Memory(this_directory/"cache", verbose=0)

# %%
# from clearml import Task
# task = Task.init(project_name="cvml detection", task_name="gen_dataset")

# %%


def read_param():
    # 读取参数
    params_file = this_directory/(this_file.name.removesuffix(".py")+".toml")
    with open(params_file, "rb") as f:
        params = tomli.load(f)
    params = DefaultMunch.fromDict(params)
    return params


def write_txt(out_file, image_items):
    """_summary_
        image_item类似于
        {'path': 'train/62627.jpg',
        'id': 62627,
        'objects': [{'bbox': {'xmin': 1580.0,
            'ymin': 758.667,
            'xmax': 1638.6667,
            'ymax': 818.6667},
        'category': 'ph5'}]}
    Args:
        out_file (_type_): 输出路径
        image_items (_type_): 图片描述
    """
    
    def item2line(image_item):
        return f'./images/{image_item.path}\n'
    image_strs = map(item2line, image_items)
    with open(out_file, 'w') as f:
        f.writelines(image_strs)
    
# %%
# @memory.cache
def handle_param(params: DefaultMunch):
    input_params = params.input
    output_params = params.output
    options_params = params.options

    # 输出参数处理
    result_path = this_directory/output_params.result_path
    result_dataset_description_path = result_path / \
        output_params.result_dataset_description_path
    result_data_location_path = result_path/output_params.result_data_location_path / \
        output_params.dataset_name  # 这里不用时间戳，防止文件夹太多
    result_dataset_description_path.mkdir(parents=True, exist_ok=True)
    result_data_location_path.mkdir(parents=True, exist_ok=True)

    dataset_name = output_params.dataset_name
    if output_params.timestamp_on_name:
        dataset_name += _current_time
    dataset_yaml_path = result_dataset_description_path/(dataset_name+".yaml")
    dataset_yaml_path = dataset_yaml_path.absolute()

    # 读取 annotations_all.json
    tt100k = this_directory / input_params.tt100k_path
    import json
    with open(tt100k/"dataset_specific"/"annotations_all.json") as f:
        annos = json.loads(f.read())
    annos = DefaultMunch.fromDict(annos)
    
    
    types = annos.types
    num2cate = dict(enumerate(annos.types))
    cate2num = {v:k for k, v in num2cate.items()}
    
    
    # txt文件生成
    train_txt_name = f"{output_params.train_txt_name}{_current_time}.txt"
    val_txt_name = f"{output_params.val_txt_name}{_current_time}.txt"
    test_txt_name = f"{output_params.test_txt_name}{_current_time}.txt"
    
    # 给txt填充内容
    imgs = list(annos.imgs.values())
    # from sklearn.model_selection import train_test_split
    from torch.utils.data import random_split
    
    train_dataset, val_dataset = random_split(
        dataset=imgs,
         lengths=[0.8, 0.2],
        generator=torch.Generator().manual_seed(0)
    )
    test_dataset = val_dataset
    
    
    write_txt(tt100k/train_txt_name, train_dataset)
    write_txt(tt100k/val_txt_name, val_dataset)
    write_txt(tt100k/test_txt_name, test_dataset)
    


    # yaml生成。 跑yolov5只需要用到这个yaml
    dataset_yaml = DefaultMunch()
    dataset_yaml.path = str(result_data_location_path.relative_to(result_path))
    dataset_yaml.train = train_txt_name
    dataset_yaml.val = val_txt_name
    dataset_yaml.test = test_txt_name
    dataset_yaml.nc = len(types)
    dataset_yaml.names = num2cate

    # 生成 dataset.yaml
    import yaml
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dict(dataset_yaml), f)


# %%
if __name__ == "__main__":
    params = read_param()
    handle_param(params)
