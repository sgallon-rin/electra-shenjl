#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2022/3/3 22:21
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : download_dataset.py
# @Description : Download dataset with datasets package
import datasets

from config import DATA_CACHE_DIR

if __name__ == "__main__":
    # openwebtext is huge
    # https://huggingface.co/datasets/openwebtext
    # dataset_name = "openwebtext"
    # datasets.load_dataset(dataset_name, cache_dir=CACHE_DIR)

    # glue benchmark
    # must specify task while downloading
    # https://huggingface.co/datasets/glue
    dataset_name = "glue"
    tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli',
             'ax']
    for task in tasks:
        datasets.load_dataset(dataset_name, task, cache_dir=DATA_CACHE_DIR)
