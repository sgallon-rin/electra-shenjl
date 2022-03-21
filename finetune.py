#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2022/3/21 11:53
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : finetune.py
# @Description :

import os
import random
import numpy as np
import torch
from torch import nn
from functools import partial
import datasets
from hugdatafast import *
from transformers import ElectraModel, ElectraConfig, ElectraTokenizerFast, ElectraForPreTraining
import neptune.new as neptune
from fastai.callback.neptune import NeptuneCallback
from utils import MyConfig, MyMSELossFlat, Adam_no_bias_correction, GradientClipping, load_model_
from config import NEPTUNE_INIT_INFO, HOME_DIR, DATA_CACHE_DIR

# # set seed
# SEED = 123
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# torch.backends.cudnn.deterministic = True

# 1. Configuration
c = MyConfig({
    'device': 'cuda:0',  # List[int]: use multi gpu (data parallel)
    # run [start,end) runs, every run finetune every GLUE tasks once with different seeds.
    'start': 0,
    'end': 10,

    'pretrained_checkpoint': None,  # None to use pretrained ++ model from HuggingFace
    # Seeds for fintuning. i th run use i th seeds, None to use system time
    # 'seeds': [939, 481, 569, 620, 159, 808, 816, 101, 554, 104], # for 11081
    # 'seeds': [611, 609, 830, 237, 668, 608, 475, 690, 53, 94], # for 36
    # 'seeds': [775, 961, 778, 915, 979, 526, 99, 669, 806, 78], # for 1188
    # 'seeds': [895, 602, 573, 457, 736, 871, 571, 84, 514, 740,], # for 76
    # 'seeds': [760, 63, 392, 240, 794, 168, 245, 345, 97, 917], # 1
    # 'seeds': [6669, 4093, 6254, 8546, 489, 901, 5567, 3690, 7057, 3663,], # for 4
    # 'seeds': [3426, 2730, 6509, 6957, 2961, 7783, 7061, 4261, 2256, 4863,], # for 4649
    # 'seeds': [3049, 3005, 3298, 8108, 6676, 2275, 376, 5053, 154, 8992,], # for 7
    'seeds': None,

    'weight_decay': 0,
    'adam_bias_correction': False,
    'xavier_reinited_outlayer': True,
    'schedule': 'original_linear',
    'original_lr_layer_decays': True,
    'double_unordered': True,

    # whether to do finetune or test
    'do_finetune': True,  # True -> do finetune ; False -> do test
    # finetuning checkpoint for testing. These will become "ckp_dir/{task}_{group_name}_{th_run}.pth"
    'th_run': {'qqp': 7, 'qnli': 5,
               'mrpc': 7, 'mnli': 2, 'ax': 2,
               'sst2': 3, 'rte': 7, 'wnli': 0,
               'cola': 1, 'stsb': 8,
               },

    'size': 'small',
    # 'size': 'base',
    # 'size': 'large',

    'num_workers': 3,
    'logger': 'neptune',
    'group_name': None,  # the name of represents these runs
    # None: use name of checkpoint.
    # False: don't do online logging and don't save checkpoints
})

# Check
if not c.do_finetune:
    assert c.th_run['mnli'] == c.th_run['ax']
assert c.schedule in ['original_linear', 'separate_linear', 'one_cycle', 'adjusted_one_cycle']

# Settings of different sizes, follow Clark et al. 2020
if c.size == 'small':
    c.lr = 3e-4
    c.layer_lr_decay = 0.8
    c.max_length = 128
elif c.size == 'base':
    c.lr = 1e-4
    c.layer_lr_decay = 0.8
    c.max_length = 512
elif c.size == 'large':
    c.lr = 5e-5
    c.layer_lr_decay = 0.9
    c.max_length = 512
else:
    raise ValueError(f"Invalid size {c.size}")
if c.pretrained_checkpoint is None:
    c.max_length = 512  # All public models is ++, which use max_length 512

# huggingface/transformers
hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{c.size}-discriminator")
electra_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-discriminator')

# logging
# light logging callback here is to only log the last score and avoid exceeding the api access limit
if c.logger == 'neptune':
    class LightNeptuneCallback(NeptuneCallback):
        def after_batch(self): pass

        def after_epoch(self):
            if self.epoch == (self.n_epoch - 1): super().after_epoch()


    neptune.init(**NEPTUNE_INIT_INFO)

# Path
CHECKPOINTS_DIR = os.path.join(HOME_DIR, "checkpoints", "glue")
OUTPUTS_DIR = os.path.join(HOME_DIR, "test_outputs", "glue")
# c.pretrained_ckp_path = Path(f'./checkpoints/pretrain/{c.pretrained_checkpoint}')
if c.group_name is None:
    # if c.pretrained_checkpoint:
    #     c.group_name = c.pretrained_checkpoint[:-4]
    # elif c.pretrained_checkpoint is None:
    #     c.group_name = f"{c.size}++"
    c.group_name = f"{c.size}++"

# Print info
print(f"process id: {os.getpid()}")
print(c)

METRICS = {
    **{task: [MatthewsCorrCoef()] for task in ['cola']},
    **{task: [accuracy] for task in ['sst2', 'mnli', 'qnli', 'rte', 'wnli', 'snli', 'ax']},
    **{task: [F1Score(), accuracy] for task in ['mrpc', 'qqp']},
    **{task: [PearsonCorrCoef(), SpearmanCorrCoef()] for task in ['stsb']}
}
NUM_CLASS = {
    **{task: 1 for task in ['stsb']},
    **{task: 2 for task in ['cola', 'sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli']},
    **{task: 3 for task in ['mnli', 'ax']},
}
TEXT_COLS = {
    **{task: ['question', 'sentence'] for task in ['qnli']},
    **{task: ['sentence1', 'sentence2'] for task in ['mrpc', 'stsb', 'wnli', 'rte']},
    **{task: ['question1', 'question2'] for task in ['qqp']},
    **{task: ['premise', 'hypothesis'] for task in ['mnli', 'ax']},
    **{task: ['sentence'] for task in ['cola', 'sst2']},
}
LOSS_FUNC = {
    **{task: CrossEntropyLossFlat() for task in ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli', 'ax']},
    **{task: MyMSELossFlat(low=0.0, high=5.0) for task in ['stsb']}
}


# 2. Data
# 2.1 Download and Preprocess
def tokenize_sents_max_len(example, cols, max_len, swap=False):
    # Follow BERT and ELECTRA, truncate the examples longer than max length
    tokens_a = hf_tokenizer.tokenize(example[cols[0]])
    tokens_b = hf_tokenizer.tokenize(example[cols[1]]) if len(cols) == 2 else []
    _max_length = max_len - 1 - len(cols)  # preserved for cls and sep tokens
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= _max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    if swap:
        tokens_a, tokens_b = tokens_b, tokens_a
    tokens = [hf_tokenizer.cls_token, *tokens_a, hf_tokenizer.sep_token]
    token_type = [0] * len(tokens)
    if tokens_b:
        tokens += [*tokens_b, hf_tokenizer.sep_token]
        token_type += [1] * (len(tokens_b) + 1)
    example['inp_ids'] = hf_tokenizer.convert_tokens_to_ids(tokens)
    example['attn_mask'] = [1] * len(tokens)
    example['token_type_ids'] = token_type
    return example


# View raw data on[nlp - viewer]! (https://huggingface.co/nlp/viewer/)
glue_dsets = {}
glue_dls = {}
for task in ['cola', 'sst2', 'mrpc', 'stsb', 'mnli', 'qqp', 'qnli', 'rte', 'wnli', 'ax']:
    # Load / download datasets.
    dsets = datasets.load_dataset('glue', task, cache_dir=DATA_CACHE_DIR)
    # There is two samples broken in QQP training set
    if task == 'qqp': dsets['train'] = dsets['train'].filter(lambda e: e['question2'] != '',
                                                             cache_file_name=os.path.join(
                                                                 dsets['train'].cache_directory(), 'fixed_train.arrow'))
    # Load / Make tokenized datasets
    tok_func = partial(tokenize_sents_max_len, cols=TEXT_COLS[task], max_len=c.max_length)
    glue_dsets[task] = dsets.my_map(tok_func, cache_file_names=f"tokenized_{c.max_length}_{{split}}")

    if c.double_unordered and task in ['mrpc', 'stsb']:
        swap_tok_func = partial(tokenize_sents_max_len, cols=TEXT_COLS[task], max_len=c.max_length, swap=True)
        swapped_train = dsets['train'].my_map(swap_tok_func,
                                              cache_file_name=f"swapped_tokenized_{c.max_length}_train")
        glue_dsets[task]['train'] = datasets.concatenate_datasets([glue_dsets[task]['train'], swapped_train])

    # Load / Make dataloaders
    hf_dsets = HF_Datasets(glue_dsets[task], hf_toker=hf_tokenizer, n_inp=3,
                           cols={'inp_ids': TensorText, 'attn_mask': noop, 'token_type_ids': noop,
                                 'label': TensorCategory})
    if c.double_unordered and task in ['mrpc', 'stsb']:
        dl_kwargs = {'train': {'cache_name': f"double_dl_{c.max_length}_train.json"}}
    else:
        dl_kwargs = None
    glue_dls[task] = hf_dsets.dataloaders(bs=32, shuffle_train=True, num_workers=c.num_workers,
                                          cache_name=f"dl_{c.max_length}_{{split}}.json",
                                          dl_kwargs=dl_kwargs, )


# 2. Finetuning
# 2.1 Finetuning model
# * ELECTRA use CLS encodings as pooled result to predict the sentence. (see [here](https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/modeling.py#L254) of its official repository)
# * Note that we should use different prediction head instance for different tasks.
class SentencePredictor(nn.Module):

    def __init__(self, model, hidden_size, num_class):
        super().__init__()
        self.base_model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_class)
        if c.xavier_reinited_outlayer:
            nn.init.xavier_uniform_(self.classifier.weight.data)
            self.classifier.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        return self.classifier(self.dropout(x[:, 0, :])).squeeze(
            -1).float()  # if regression task, squeeze to (B), else (B,#class)


# 2.2 Discriminative learning rate
def list_parameters(model, submod_name):
    return list(eval(f"model.{submod_name}").parameters())


def hf_electra_param_splitter(model):
    base = 'base_model'
    embed_name = 'embeddings'
    scaler_name = 'embeddings_project'
    layers_name = 'layer'
    output_name = 'discriminator.discriminator_predictions'

    groups = [list_parameters(model, f"{base}.{embed_name}")]
    for i in range(electra_config.num_hidden_layers):
        groups.append(list_parameters(model, f"{base}.encoder.{layers_name}[{i}]"))
    groups.append(list_parameters(model, output_name))
    if electra_config.hidden_size != electra_config.embedding_size:
        groups[0] += list_parameters(model, f"{base}.{scaler_name}")

    assert len(list(model.parameters())) == sum([len(g) for g in groups])
    for i, (p1, p2) in enumerate(zip(model.parameters(), [p for g in groups for p in g])):
        assert torch.equal(p1, p2), f"The {i} th tensor"
    return groups


def get_layer_lrs(lr, decay_rate, num_hidden_layers):
    lrs = [lr * (decay_rate ** depth) for depth in range(num_hidden_layers + 2)]
    if c.original_lr_layer_decays:
        for i in range(1, len(lrs)): lrs[i] *= decay_rate
    return list(reversed(lrs))


## 2.3 learner
def get_glue_learner(i, task, run_name=None, inference=False):
    # Num_epochs
    if task in ['rte', 'stsb']:
        num_epochs = 10
    else:
        num_epochs = 3

    # Dataloaders
    dls = glue_dls[task]
    if isinstance(c.device, str):
        dls.to(torch.device(c.device))
    elif isinstance(c.device, list):
        dls.to(torch.device('cuda', c.device[0]))
    else:
        dls.to(torch.device('cuda:0'))

    # Load pretrained model
    # use pretrained ++ model from HuggingFace
    discriminator = ElectraForPreTraining.from_pretrained(f"google/electra-{c.size}-discriminator")

    # Seeds & PyTorch benchmark
    torch.backends.cudnn.benchmark = True
    if c.seeds:
        dls[0].rng = random.Random(c.seeds[i])  # for fastai dataloader
        random.seed(c.seeds[i])
        np.random.seed(c.seeds[i])
        torch.manual_seed(c.seeds[i])

    # Create finetuning model
    model = SentencePredictor(discriminator.electra, electra_config.hidden_size, num_class=NUM_CLASS[task])

    # Discriminative learning rates
    splitter = hf_electra_param_splitter
    layer_lrs = get_layer_lrs(lr=c.lr,
                              decay_rate=c.layer_lr_decay,
                              num_hidden_layers=electra_config.num_hidden_layers, )

    # Optimizer
    if c.adam_bias_correction:
        opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=c.weight_decay)
    else:
        opt_func = partial(Adam_no_bias_correction, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=c.weight_decay)

    # Learner
    learn = Learner(dls, model,
                    loss_func=LOSS_FUNC[task],
                    opt_func=opt_func,
                    metrics=METRICS[task],
                    splitter=splitter if not inference else trainable_params,
                    lr=layer_lrs if not inference else defaults.lr,
                    path=CHECKPOINTS_DIR,
                    model_dir=c.group_name, )

    # Multi gpu
    if isinstance(c.device, list) or c.device is None:
        learn.create_opt()
        learn.model = nn.DataParallel(learn.model, device_ids=c.device)

    # Mixed precision
    learn.to_native_fp16(init_scale=2. ** 14)

    # Gradient clip
    learn.add_cb(GradientClipping(1.0))

    # Logging
    # Logging
    if run_name and not inference:
        if c.logger == 'neptune':
            neptune.create_experiment(name=run_name, params={'task': task, **c, **hparam_update})
            learn.add_cb(LightNeptuneCallback(False))

    # Learning rate schedule
    if c.schedule == 'one_cycle':
        return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs, lr_max=layer_lrs)
    elif c.schedule == 'adjusted_one_cycle':
        return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs, lr_max=layer_lrs, div=1e5, pct_start=0.1)
    else:
        lr_shed_func = linear_warmup_and_then_decay if c.schedule == 'separate_linear' else linear_warmup_and_decay
        lr_shedule = ParamScheduler({'lr': partial(lr_shed_func,
                                                   lr_max=np.array(layer_lrs),
                                                   warmup_pct=0.1,
                                                   total_steps=num_epochs * (len(dls.train)))})
        return learn, partial(learn.fit, n_epoch=num_epochs, cbs=[lr_shedule])


#  2.4 Do finetuning
if c.do_finetune:
    for i in range(c.start, c.end):
        for task in ['cola', 'sst2', 'mrpc', 'stsb', 'rte', 'qnli', 'qqp', 'mnli', 'wnli']:
            if c.group_name:
                run_name = f"{c.group_name}_{task}_{i}"
            else:
                run_name = None
                print(task)
            learn, fit_fc = get_glue_learner(i, task, run_name)
            fit_fc()
            if run_name: learn.save(f"{task}_{i}")

# 3. Testing
# Haven't found way to validate and log two datasets in the training loop, so validate mnli-mm here as a workaround
if not c.do_finetune:
    learn, _ = get_glue_learner('mnli', inference=True)
    learn.load(f"mnli_{c.th_run['mnli']}")
    with learn.no_mbar():
        print(learn.validate(ds_idx=2))


def get_identifier(task, split):
    "Turn task name to official task identifier defined."
    map = {'cola': 'CoLA', 'sst2': 'SST-2', 'mrpc': 'MRPC', 'qqp': 'QQP', 'stsb': 'STS-B', 'qnli': 'QNLI', 'rte': 'RTE',
           'wnli': 'WNLI', 'ax': 'AX'}
    if task == 'mnli' and split == 'test_matched':
        return 'MNLI-m'
    elif task == 'mnli' and split == 'test_mismatched':
        return 'MNLI-mm'
    else:
        return map[task]


def predict_test(task, checkpoint, dl_idx):
    output_dir = Path(f'{OUTPUTS_DIR}/{c.group_name}')
    output_dir.mkdir(exist_ok=True)
    device = torch.device(c.device)

    # Load checkpoint and get predictions
    learn, _ = get_glue_learner(task, inference=True)
    load_model_(learn, checkpoint)
    results = learn.get_preds(dl=learn.dls[dl_idx], with_decoded=True)
    preds = results[-1]  # preds -> (predictions logits, targets, decoded prediction)

    # Decode target class index to its class name
    if task in ['mnli', 'ax']:
        preds = [['entailment', 'neutral', 'contradiction'][p] for p in preds]
    elif task in ['qnli', 'rte']:
        preds = [['entailment', 'not_entailment'][p] for p in preds]
    elif task == 'wnli' and c.wsc_trick:
        preds = preds.to(dtype=torch.long).tolist()
    else:
        preds = preds.tolist()

    # Form test dataframe and save
    test_df = pd.DataFrame({'index': range(len(list(glue_dsets[task].values())[dl_idx])), 'prediction': preds})
    split = list(glue_dsets['mnli'].keys())[dl_idx] if task == 'mnli' else 'test'
    identifier = get_identifier(task, split)
    test_df.to_csv(output_dir / f'{identifier}.tsv', sep='\t')
    return test_df


if not c.do_finetune:
    for task, th in c.th_run.items():
        print(task)
        # ax use mnli ckp
        if isinstance(th, int):
            ckp = f"{task}_{th}" if task != 'ax' else f"mnli_{th}"
        else:
            ckp = [f"{task}_{i}" if task != 'ax' else f"mnli_{i}" for i in th]
        # run test for all testset in this task
        dl_idxs = [-1, -2] if task == 'mnli' else [-1]
        for dl_idx in dl_idxs:
            df = predict_test(task, ckp, dl_idx)
