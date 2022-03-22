# Deep learning group study
2022.03.22

## Finetune ELECTRA on GLUE

I referred to implementations from
https://github.com/richarddwang/electra_pytorch
and
https://github.com/guevarsd/GLUE-Benchmark-NLP

### Major packages used
```
transformers
datasets
wandb
```

### Usage

To fine-tune on a task `task_name` ("cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli") 
with ELECTRA model size `model_size` ("small", "base", "large") 
```
$ python finetune_glue.py -t task_name -s model_size
```

To run test
```
$ python test_glue.py -t task_name -s model_size
```

## Tips

### Packages `transformers` and `datasets`

#### Cache management

To change the default cache directory for `transformers` (refer
to https://huggingface.co/docs/transformers/installation):

```
$ export TRANSFORMERS_CACHE="/raid_elmo/home/lr/shenjl/cache/huggingface/transformers"
```

To change the default cache directory for `dataset` (refer to https://huggingface.co/docs/datasets/cache):

```
export HF_DATASETS_CACHE="/raid_elmo/home/lr/shenjl/cache/huggingface/datasets"
```

#### About `datasets.load_dataset`

If you want to load a dataset from downloaded cache, please make sure there is not a directory with the same name as the
dataset (e.g. "glue") in the working directory.

Otherwise, `datasets.load_dataset` will try to load from the local directory and may cause error like:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/lr/shenjl/anaconda3/envs/electra/lib/python3.8/site-packages/datasets/load.py", line 1675, in load_dataset
    builder_instance = load_dataset_builder(
  File "/home/lr/shenjl/anaconda3/envs/electra/lib/python3.8/site-packages/datasets/load.py", line 1512, in load_dataset_builder
    dataset_module = dataset_module_factory(
  File "/home/lr/shenjl/anaconda3/envs/electra/lib/python3.8/site-packages/datasets/load.py", line 1133, in dataset_module_factory
    return LocalDatasetModuleFactoryWithoutScript(
  File "/home/lr/shenjl/anaconda3/envs/electra/lib/python3.8/site-packages/datasets/load.py", line 726, in get_module
    data_files = DataFilesDict.from_local_or_remote(
  File "/home/lr/shenjl/anaconda3/envs/electra/lib/python3.8/site-packages/datasets/data_files.py", line 578, in from_local_or_remote
    DataFilesList.from_local_or_remote(
  File "/home/lr/shenjl/anaconda3/envs/electra/lib/python3.8/site-packages/datasets/data_files.py", line 546, in from_local_or_remote
    data_files = resolve_patterns_locally_or_by_urls(base_path, patterns, allowed_extensions)
  File "/home/lr/shenjl/anaconda3/envs/electra/lib/python3.8/site-packages/datasets/data_files.py", line 203, in resolve_patterns_locally_or_by_urls
    raise FileNotFoundError(error_msg)
FileNotFoundError: Unable to resolve any data file that matches '['**train*']' at /raid_elmo/home/lr/shenjl/data/glue with any supported extension ['csv', 'tsv', 'json', 'jsonl', 'parquet', 'txt', 'zip']
```

### Online logging with `wandb`

Refer to https://wandb.ai for details.