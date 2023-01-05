# 输入
# 数据集
# 输出
# 转换：1. xyxy->xywh 2. label name -> label id
# %%
from concurrent import futures
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
from tqdm import tqdm
this_file = Path(__file__).absolute()
this_directory = this_file.parent
_current_time = f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

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

# %%


def xyxy2xywh(size, box):
    # 比如， size = （12, 13）
    # box = 'xmin': 1580.0, 'ymin': 758.667, 'xmax': 1638.6667, 'ymax': 818.6667
    # converts to yolo format
    w_scale = size[0]
    h_scale = size[1]
    x = (box[0] + box[2])/2.0  # xmin+xmax
    y = (box[1] + box[3])/2.0  # ymin+ymax
    w = box[2] - box[0]  # xmax-xmin
    h = box[3] - box[1]  # ymax-ymin
    x /= w_scale
    w /= w_scale
    y /= h_scale
    h /= h_scale
    return (x, y, w, h)


# @memory.cache


def generate_label_txt(image_file, image_name, objects, label_directory, cate2num, override=False):
    """生成image_name.txt 每一行代表一个标注。
        1. xyxy->xywh 2. label name -> label id
    Args:
        image_file (str):  图片的路径，比如 images/train/12.jpg
        image_name (str): 图片的名字，比如12
        objects (array): bbox的数组。
        label_directory (_type_): 标签的路径，比如 labels/train
        cate2num (dict): 比如{'pl80':0}
    """
    out_file = Path(label_directory)/f"{image_name}.txt"
    # if override or (not out_file.exists()):
    img_size = imagesize.get(str(image_file))
    def object2line(object):
        bbox = object.bbox # 不能假设python3.9字典就有序。
        x, y, w, h = xyxy2xywh(img_size, (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax))
        # return f"{cate2num[object.category]} {x} {y} {w} {h}"
        return f"{cate2num[object.category]} {x} {y} {w} {h}\n" #必须换行！
    box_strs = map(object2line, objects)
    with open(out_file, 'w') as f:
        f.writelines(box_strs)


def generate_label_txt_by_item(item, tt100k, image_directory, label_directory, cate2num, override=False):
    """ 根据类似于如下的item对其进行标注
        {'path': 'train/62627.jpg',
        'id': 62627,
        'objects': [{'bbox': {'xmin': 1580.0,
            'ymin': 758.667,
            'xmax': 1638.6667,
            'ymax': 818.6667},
        'category': 'ph5'}]}
    Args:
        item (_type_): _description_
        label_directory (_type_): 标签的路径，比如 labels/train
        cate2num (dict): 比如{'pl80':0}
    """
    prefix = "train" if item.path.startswith("train") else "test"
    image_file = image_directory/item.path
    return generate_label_txt(image_file, str(item.id), item.objects, label_directory/prefix, cate2num, override=False)

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
    
    # 生成labels
    img_val = list(annos.imgs.values())
    num2cate = dict(enumerate(annos.types))
    cate2num = {v:k for k, v in num2cate.items()}
    
    image_directory, label_directory = tt100k/"images", tt100k/"labels"
    
    tasks = []
    with futures.ThreadPoolExecutor(max_workers=128) as executor:
        # 遍历所有的正样本。注意不是遍历train_path_pos，而是glob的结果——positive_files。
        for i, item in enumerate(img_val):
            # 多线程
            tasks.append(
                executor.submit(
                    generate_label_txt_by_item,
                    item, tt100k,image_directory, label_directory,  cate2num, False
                )
            )
            # # 单线程
            # generate_label_txt_by_item(
            #         item, tt100k,image_directory, label_directory,  cate2num, False)
        for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
            pass  # 等待所有任务完成
        
    
    



# %%
if __name__ == "__main__":
    params = read_param()
    handle_param(params)
