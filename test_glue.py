#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2022/3/21 20:48
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : test_glue.py
# @Description : Test Fine-tuned ELECTRA on GLUE tasks
# based on https://github.com/guevarsd/GLUE-Benchmark-NLP/blob/main/Code/EvaluateLoop.py

import os
import sys
import getopt
import time
import random
import torch
import numpy as np
from transformers import ElectraTokenizerFast
from transformers import ElectraForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import wandb
from config import HOME_DIR, DATA_CACHE_DIR, METRIC_CACHE_DIR, TEST_RESULT_FILE

# Commandline params for task and model
argv = sys.argv[1:]
task = ""
size = ""
try:
    opts, args = getopt.getopt(argv, "t:s:", ["task=", "size="])
except:
    raise ValueError("Task and size unspecified!")
for opt, arg in opts:
    if opt in ['-t', '--task']:
        task = arg
    elif opt in ['-s', '--size']:
        size = arg

# Set seed
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# List of glue tasks
GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

# Pretrained model sizes
MODEL_SIZES = ["small", "base", "large"]
MODEL_LR = {"small": 3e-4, "base": 1e-4, "large": 5e-5}

# Path
CHECKPOINTS_DIR = os.path.join(HOME_DIR, "checkpoints", "glue")
# OUTPUTS_DIR = os.path.join(HOME_DIR, "test_outputs", "glue")

# Configuration
assert size in MODEL_SIZES, "Size must be in {}".format(MODEL_SIZES)
assert task in GLUE_TASKS, "Task must be in {}".format(GLUE_TASKS)

model_checkpoint = "google/electra-{}-discriminator".format(size)
batch_size = 32  # follow ELECTRA

# Verify baseline already established
ckpt_path = os.path.join(CHECKPOINTS_DIR, model_checkpoint + "_tuned_" + task)

# if ckpt_path not in os.listdir():
#     raise ValueError('Checkpoint not found: {}. Stopping Process.'.format(ckpt_path))

# wandb
# wandb_config = {"task": task + "_eval", "size": size, "ckpt_path": ckpt_path}
# wandb.init(project="electra-shenjl", entity="sgallon-rin", config=wandb_config)

# Load dataset based on task variable
dataset = load_dataset("glue", task, cache_dir=DATA_CACHE_DIR)
metric = load_metric('glue', task, cache_dir=METRIC_CACHE_DIR)

# Create tokenizer for respective model
tokenizer = ElectraTokenizerFast.from_pretrained(model_checkpoint, use_fast=True, model_max_length=512, truncation=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# select columns with sentences to tokenize based on given task
sentence1_key, sentence2_key = task_to_keys[task]


def tokenizer_func(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


# tokenize sentence(s)
encoded_dataset = dataset.map(tokenizer_func, batched=True)

# encoded_dataset = encoded_dataset.remove_columns(column_names=["idx", "premise", "hypothesis"])
# encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "question", "sentence"])
# encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "sentence1", "sentence2"])
# encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "sentence"])


# Number of logits to output
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

# Create model and attach ForSequenceClassification head
model = ElectraForSequenceClassification.from_pretrained(ckpt_path, num_labels=num_labels)

# Type of metric for given task
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

args = TrainingArguments(
    output_dir=os.path.join(CHECKPOINTS_DIR, f"{model_checkpoint}-finetuned-test-{task}"),
    evaluation_strategy="epoch",
    weight_decay=0,
    per_device_eval_batch_size=batch_size,
    metric_for_best_model=metric_name,
    eval_accumulation_steps=5,
    # report_to="wandb"
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

score = trainer.evaluate()

# show and save score
localtime = time.asctime(time.localtime(time.time()))
res = "Time: {}\nTask: {}\nModel size: {}\nMetric name: {}\nTest metric value: {}\n\n".format(localtime, task, size,
                                                                                              metric_name, score)
print(res)
with open(TEST_RESULT_FILE, "a+") as handler:
    handler.write(res)
