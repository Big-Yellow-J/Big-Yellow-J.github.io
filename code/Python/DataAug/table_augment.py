import os
import json
from tqdm import tqdm
from collections import defaultdict

def open_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class DataBuildDPODataset():
    def __init__(self, json_path, image_path=None, random_ratio= [0.2, 0.4, 0.6, 0.8]):
        self.json_path = json_path
        self.image_path = image_path

        self.json_file = open_json(json_path)
    
    def DPO_augment_table(self):
        '''
        对于html表格进行数据增强处理，主要用于构建DPO的数据集
        强化处理操作包括：1、随机行列交换；2、随机生成/取消部分线条；3、随机合并/拆分部分表格；4、随机给部分表格上“标记”
        处理流程：先去增强处理、最后格式化处理
        '''
        
        pass

if __name__ == '__main__':
    data = open_json('./pubtabnet.json')
    result = defaultdict(list)
    for i, key in enumerate(data.keys()):
        if i<=5:
            result[key].append(data[key])
    with open('./tmp_use.json', 'w', encoding= 'utf-8') as f:
        json.dump(result, f, ensure_ascii= True, indent=2)