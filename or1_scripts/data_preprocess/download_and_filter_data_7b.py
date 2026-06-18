# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format

中文学习提示：
这个脚本是 Skywork-OR1 训练数据准备链路的入口之一。它并不是从头构造题目，
而是从 Hugging Face 数据集 Skywork/Skywork-OR1-RL-Data 下载官方已经整理好的
数学和代码 RL 数据，再按“某个基座模型觉得这道题有多难”进行过滤。

面试讲法：
- RL 训练不希望题目太简单，因为所有采样都答对时没有区分度。
- 也不希望题目太难，因为所有采样都答错时同样没有训练信号。
- 这里用 model_difficulty 过滤出适合 7B 基座模型学习的题目。
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./or1_data/train')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 官方发布的 RL 训练数据。数据里每条样本通常包含：
    # prompt: 聊天格式的问题；
    # ability: math 或 code；
    # reward_model.ground_truth: verifier 需要的标准答案或测试用例；
    # extra_info.model_difficulty: 不同模型上的难度估计。
    data_source = 'Skywork/Skywork-OR1-RL-Data'

    dataset = datasets.load_dataset(data_source)
    print("\n\nOriginal dataset: ", "\n", dataset)

    def process_ground_truth(item):
        # 某些 ground_truth 在数据集中以 JSON 字符串保存。
        # 这里尽量解析成 Python dict/list，方便后续 reward function 直接使用。
        if "reward_model" in item and "ground_truth" in item["reward_model"]:
            try:
                item["reward_model"]["ground_truth"] = json.loads(item["reward_model"]["ground_truth"])
            except:
                pass
        return item
    
    dataset= dataset.map(process_ground_truth)
    
    def filter_fn(example):  
        # 只保留带有模型难度标注的样本。
        # 对 7B 模型而言，difficulty 1-15 被认为是可学习区间：
        # 太简单/太难的题目都会让 GRPO 的同题多采样奖励缺少方差。
        if 'extra_info' not in example or 'model_difficulty' not in example['extra_info']:
            return False 
        difficulty = example['extra_info']['model_difficulty'].get('DeepSeek-R1-Distill-Qwen-7B')
        if difficulty is None:
            return False
        if difficulty < 1 or difficulty > 15:
            return False
        return True
    dataset = dataset.filter(filter_fn)
    print("\n\nFiltered dataset: ", "\n", dataset)

    data_list = []
    for key in dataset:
        data_list.extend([item for item in dataset[key]])

    # Skywork-OR1 同时训练数学推理和代码推理。
    # 后续训练脚本会把 math/code pkl 混合起来喂给同一个 PPO/GRPO 训练器。
    math_data_list = [item for item in data_list if item['ability'] == 'math']
    code_data_list = [item for item in data_list if item['ability'] == 'code']
    
    for i in range(len(code_data_list)):
        # 代码题的 ground_truth 可能包含 None 字段。
        # verifier 执行单测时只需要有效测试字段，因此这里清理掉空值。
        new_ground_truth = {}
        item = code_data_list[i]['reward_model']['ground_truth']
        for key in item:
            if item[key] is not None:
                new_ground_truth[key] = item[key]
        code_data_list[i]['reward_model']['ground_truth'] = new_ground_truth

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)
    # 输出是 pickle 文件，不是 parquet。RLHFDataset 里兼容 .pkl 和 .parquet 两种读取方式。
    with open(os.path.join(local_dir, 'train_7b_math.pkl'), 'wb') as f:
        pickle.dump(math_data_list, f)
    with open(os.path.join(local_dir, 'train_7b_code.pkl'), 'wb') as f:
        pickle.dump(code_data_list, f)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
