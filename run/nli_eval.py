import datasets
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser

import os
import json


# source: https://github.com/gregdurrett/fp-dataset-artifacts
import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
from tqdm.auto import tqdm


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=128):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length
    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )
    tokenized_examples['label'] = examples['label']
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


def list_dir(path):
    contents = os.listdir(path)
    directories = [os.path.join(path, content) for content in contents if os.path.isdir(os.path.join(path, content))]
    return directories


NUM_PREPROCESSING_WORKERS = 16
os.environ["WANDB_DISABLED"] = "true"



parser = argparse.ArgumentParser(description = "NLI data augmentation with nlpaug...)",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--val_data", required = True, help = "json data input, file need to be reable by hugging face dataset method")
parser.add_argument("--test_data", required = True, help = "json data input, file need to be reable by hugging face dataset method")
parser.add_argument('--model_path', type=str, required=True, help= "path to training model with all checkpoints")
parser.add_argument('--out', type=str, default = 'out.txt', help= "output performances")



args = parser.parse_args()



# cmd = ['--model_path', '/mnt/SCRATCH/ngda/nlp/run_project/all_rm_aug_model', '--val_data', '/mnt/SCRATCH/ngda/nlp/run_project/val_no_premise.json', '--test_data', '/mnt/SCRATCH/ngda/nlp/run_project/val_no_premise.json']

# args = parser.parse_args(cmd)

val_data = datasets.load_dataset('json', data_files=args.val_data)['train']
test_data = datasets.load_dataset('json', data_files=args.test_data)['train']
all_check_points = list_dir(args.model_path)

check_point = all_check_points[-1]

eval_kwargs = {}
task_kwargs = {'num_labels': 3}
compute_metrics = compute_accuracy

eval_predictions = None
def compute_metrics_and_store_predictions(eval_preds):
    #nonlocal eval_predictions
    eval_predictions = eval_preds
    return compute_metrics(eval_preds)

model_class = AutoModelForSequenceClassification
model = model_class.from_pretrained(check_point, **task_kwargs)
tokenizer = AutoTokenizer.from_pretrained(check_point, use_fast=True)

prepare_train_dataset = prepare_eval_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, 128)

val_data_featurized = val_data.map(
    prepare_eval_dataset,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    remove_columns=val_data.column_names
)

test_data_featurized = test_data.map(
    prepare_eval_dataset,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    remove_columns=test_data.column_names
)

# trainer_class = Trainer
# trainer = trainer_class(
#     model=model,
#     eval_dataset=val_data_featurized,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics_and_store_predictions
# )

def eval_check_point(check_point, data_featurized):
    model = model_class.from_pretrained(check_point, **task_kwargs)
    trainer_class = Trainer
    trainer = trainer_class(
        model=model,
        eval_dataset=data_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    results = trainer.evaluate(**eval_kwargs)
    return results['eval_accuracy']


validation_results = []
for i, c in enumerate(all_check_points):
    try:
        tem = eval_check_point(c, val_data_featurized)
        validation_results.append(tem)
        print(f'checkpoint {i}/{len(all_check_points)}-{c}, accuracy: {tem}')
    except:
        pass


# best checkpoint
best = np.argmax(validation_results)

print(f'The best performance in validation set is {validation_results[best]}')
print(f'The best checkpoint in validation set is {all_check_points[best]}')

# compute final accuracy
res = eval_check_point(all_check_points[best], test_data_featurized)
print(f'final TEST set performance is {res}')

with open(args.out, encoding='utf-8', mode='w') as f:
    f.write(f'The best performance in validation set is {validation_results[best]}\n')
    f.write(f'The best checkpoint in validation set is {all_check_points[best]}\n')
    f.write(f'final TEST set performance is {res}\n')

