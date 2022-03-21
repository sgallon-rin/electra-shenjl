#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2022/3/21 20:48
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : finetune_glue.py
# @Description : Fine-tune ELECTRA on GLUE
# based on https://github.com/guevarsd/GLUE-Benchmark-NLP/blob/main/Code/SingleModelTraining.py

import os
import random
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from config import HOME_DIR, DATA_CACHE_DIR, METRIC_CACHE_DIR

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
OUTPUTS_DIR = os.path.join(HOME_DIR, "test_outputs", "glue")

# Configuration
c = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'batch_size': 10,
    # Vanilla ELECTRA settings
    'weight_decay': 0,
    'adam_bias_correction': False,
    'xavier_reinited_outlayer': True,
    'schedule': 'original_linear',
    'original_lr_layer_decays': True,
    'double_unordered': True,
    # whether to do finetune or test
    'do_finetune': True,  # True -> do finetune ; False -> do test
    # pretrained electra model size
    'size': 'small',
    # 'size': 'base',
    # 'size': 'large',
    'num_workers': 3,
    'logger': 'neptune',
    # task to fine-tune on
    "task": "cola"
}

assert c["size"] in MODEL_SIZES, "Size must be in {}".format(MODEL_SIZES)
assert c["task"] in GLUE_TASKS, "Task must be in {}".format(GLUE_TASKS)

task = c["task"]
model_checkpoint = "google/electra-{}-discriminator".format(c["size"])
batch_size = c["batch_size"]  # 10 normally, 8 for qnli

# Verify baseline not already established
ckpt_path = os.path.join(CHECKPOINTS_DIR, model_checkpoint + "_tuned_" + task)

if ckpt_path in os.listdir():
    response = input(f'Tuning for {task} already established. Continue? [y/n]')
    if response.lower() in ['yes', 'y']:
        print('Continuing.')
    else:
        raise ValueError('Stopping Process.')

# Load dataset based on task variable
dataset = load_dataset("glue", task, cache_dir=DATA_CACHE_DIR)
metric = load_metric('glue', task, cache_dir=METRIC_CACHE_DIR)

# Create tokenizer for respective model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_length=512, truncation=True)

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

encoded_dataset = encoded_dataset.remove_columns(column_names=["idx", "premise", "hypothesis"])
# encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "question", "sentence"])
# encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "sentence1", "sentence2"])
# encoded_dataset=encoded_dataset.remove_columns(column_names=["idx", "sentence"])


# Number of logits to output
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

# Create model and attach ForSequenceClassification head
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Type of metric for given task
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

args = TrainingArguments(
    f"{model_checkpoint}-finetuned-{task}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=MODEL_LR[c["size"]],
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    eval_accumulation_steps=5
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
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Save Model
model.save_pretrained(ckpt_path)
print("Pretrianed model saved to {}".format(ckpt_path))
