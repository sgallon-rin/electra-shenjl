#!/bin/bash
# finetune ELECTRA small/base/large on all GLUE tasks
glue_tasks_all=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")

# some tasks have already been trained, this is just for my own usage
glue_tasks_small=("mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")
for task in ${glue_tasks_small[*]}
do
  echo "Start finetuning for size-small task-${task}"
  python finetune_glue.py -t "${task}" -s small
  echo "Finished finetuning for size-small task-${task}"
done

glue_tasks_base=("mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")
for task in ${glue_tasks_base[*]}
do
  echo "Start finetuning for size-base task-${task}"
  python finetune_glue.py -t "${task}" -s base
  echo "Finished finetuning for size-base task-${task}"
done

glue_tasks_large=("mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")
for task in ${glue_tasks_large[*]}
do
  echo "Start finetuning for size-large task-${task}"
  python finetune_glue.py -t "${task}" -s large
  echo "Finished finetuning for size-large task-${task}"
done