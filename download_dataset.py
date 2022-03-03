#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2022/3/3 22:21
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : download_dataset.py
# @Description : Download dataset with datasets package
import datasets


def download(dataset_name="openwebtext"):
    datasets.load_dataset(dataset_name)


if __name__ == "__main__":
    dataset_name = "openwebtext"
    print("Downloading dataset {}".format(dataset_name))
    download(dataset_name)
    print("Dataset successfully downloaded!")
