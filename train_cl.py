import os
import argparse
import pickle
import debugpy
import torch
import torch.nn as nn
import random
import numpy as np
import datasets
import datetime
import pandas as pd
from tqdm import tqdm
from typing import Dict
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging
from transformers import default_data_collator
from transformers.optimization import get_scheduler, AdamW
#from torch.optim import AdamW

from transformers.trainer_utils import PredictionOutput, denumpify_detensorize
from transformers.trainer_pt_utils import get_parameter_names, nested_detach, nested_numpify

from model import BertForContrastiveLearning, RobertaForContrastiveLearning
#from generate import TokenFrequency, wda_by_copying, save_da_to_tsv

def _secs2timedelta(secs):
    """
    convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimals
    """
    msec = int(abs(secs - int(secs)) * 100)
    return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"

def metrics_format(metrics: Dict[str, float]) -> Dict[str, float]:
    metrics_copy = metrics.copy()
    for k, v in metrics_copy.items():
        if "_mem_" in k:
            metrics_copy[k] = f"{ v >> 20 }MB"
        elif "_runtime" in k:
            metrics_copy[k] = _secs2timedelta(v)
        elif k == "total_flos":
            metrics_copy[k] = f"{ int(v) >> 30 }GF"
        elif type(metrics_copy[k]) == float:
            metrics_copy[k] = round(v, 4)

    return metrics_copy

def log_metrics(split, metrics):
    print(f"***** {split} metrics *****")
    metrics_formatted = metrics_format(metrics)
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    v_width = max(len(str(x)) for x in metrics_formatted.values())
    for key in sorted(metrics_formatted.keys()):
        print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")


#==============================================================================#
# my classes and functions                                                     #
#==============================================================================#
class RawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, texts=None):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts

    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class NLIDataset(RawDataset):
    def __init__(self, encodings, labels):
        super().__init__()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def load_pretrained_model(args, filename=None):
    #config = AutoConfig.from_pretrained(args.model, num_labels=3, finetuning_task=args.task_name)
    config = AutoConfig.from_pretrained(args.model, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    if args.model == 'bert-base-uncased':
        model = BertForContrastiveLearning.from_pretrained(args.model, config=config)
    elif args.model == 'roberta-base':
        model = RobertaForContrastiveLearning.from_pretrained(args.model, config=config)
    else:
        raise NotImplementedError
    
    if filename is not None:
        model.load_state_dict(torch.load(filename))
    return config, tokenizer, model

def load_nli_dataset_tsv(args, tokenizer=None, filename=None, pos_a=0, pos_b=1):
    if filename is None:
        print("dataset file name not specified!")
        exit(0)
    with open(filename, 'r', encoding='utf-8') as f:
        lines = []
        while True:
            line = f.readline()
            if not line: break
            line = line.rstrip().split('\t')
            lines.append(line)
    
    texts = []
    labels = []
    for i in range(1, len(lines)):
        line = lines[i]
        texts.append([line[pos_a], line[pos_b]])
        if line[-1] == 'contradiction': labels.append(2)
        elif line[-1] == 'entailment': labels.append(0)
        elif line[-1] == 'neutral': labels.append(1)
        else:
            ValueError('label not known')

    if tokenizer is None:
        print("tokenizer must not be None")
        exit(0)
    encodings = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    return NLIDataset(encodings, labels)

def load_dataset_from_file(filename, tokenizer=None, file_type='tsv', pos_s1=8, pos_s2=9, pos_lb=-1, label_list=None, task_to_key=('sentence1', 'sentence2'), return_type='RawDataset', is_sep=True):
    # Preprocessing the raw_datasets
    if task_to_key is not None:
        sentence1_key, sentence2_key = task_to_key
    else:
        raise ValueError("Define task_to_key.")

    texts, labels = [], []
    if file_type == 'tsv':
        # read dataset
        with open(filename, 'r', encoding='utf-8') as f:
            lines = []
            while True:
                line = f.readline()
                if not line: break
                line = line.rstrip().split('\t')
                lines.append(line)

        for i in range(1, len(lines)):
            line = lines[i]
            if pos_s2 is None:
                text = line[pos_s1]
            elif not is_sep:
                text = line[pos_s1] + ' ' + line[pos_s2]
            else:
                text = [line[pos_s1], line[pos_s2]]
            texts.append(text)

            label = line[pos_lb].lower()
            if label in label_list:
                labels.append(label_list.index(label))
            else:
                ValueError('label not known')
                
    elif file_type == 'pkl':
        raw_dataset = pickle.load(open(filename, 'rb'))
        
        for i, ex in enumerate(raw_dataset):
            text = [ex[sentence1_key], ex[sentence2_key]] if sentence2_key is not None else ex[sentence1_key]
            texts.append(text)

            if ex['label'] in label_list:
                labels.append(label_list.index(ex['label']))
            else:
                ValueError('label not known')
                    
    print(f"> Num examples = {len(labels)}")

    encodings = None
    if tokenizer is not None:
        encodings = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        
    if return_type == 'RawDataset':
        raw_dataset = RawDataset(encodings, labels, texts)
    elif tokenizer is None:
        raw_dataset = (texts, labels)
    else:
        raw_dataset = (texts, labels, encodings)
    
    return raw_dataset

def texts_to_dataset(texts, labels=None, tokenizer=None, return_type='RawDataset'):
    encodings = None
    if tokenizer is not None:
        encodings = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        
    if return_type == 'RawDataset':
        raw_dataset = RawDataset(encodings, labels, texts)
    elif tokenizer is None:
        raw_dataset = (texts, labels)
    else:
        raw_dataset = (texts, labels, encodings)
        
    return raw_dataset

def combine_datasets(datasets):
    main_set = datasets[0]
    
    for key in main_set.encodings.keys():
        all_encodings = [d.encodings[key] for d in datasets]
        main_set.encodings[key] = torch.stack(all_encodings, dim=1)

    return main_set

def create_dataloader(args, dataset, sample='sequential'):
    sampler = None
    if sample == 'random':
        sampler = RandomSampler(dataset)
    elif sample == 'sequential':
        sampler = SequentialSampler(dataset)
    else:
        print("invalid sampler")
        exit(0)

    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=default_data_collator, pin_memory=True)

def create_optimizer(args, model):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "betas": (0.9, 0.999),
        "eps": 1e-08,
    }
    optimizer_kwargs["lr"] = args.learning_rate
    return AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

def train(args, model, dataset):
    
    dataloader = create_dataloader(args, dataset, sample='random')
    optimizer = create_optimizer(args, model)
    num_training_steps = args.num_epochs * len(dataloader)
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=50, num_training_steps=num_training_steps)####here

    tr_loss = torch.tensor(0.0).to(args.device)
    model.zero_grad()

    for _ in range(args.num_epochs):

        for inputs in tqdm(dataloader):
            
            # training_step ----------------------------------------------------
            model.train() # train mode

            # _prepare_inputs --------------------------------------------------
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)

            # original model ---------------------------------------------------
            labels = None
            outputs = model(**inputs) # forward
            loss = outputs["loss"]
            
            # compute_loss -----------------------------------------------------
            loss.backward()
            tr_loss += loss.detach()

            # gradient clipping (max_grad_norm = 1.0) --------------------------
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()

    return model

def evaluate(args, model, dataset, model_path=None, output_dir='tsne/encoder_output/', filename=None):
    test_dataloader = create_dataloader(args, dataset)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    losses_host = None
    preds_host = None
    labels_host = None

    all_losses = None
    all_preds = None
    all_labels = None
    
    all_encoder_outputs = None

    # evaluation loop ----------------------------------------------------------
    for inputs in tqdm(test_dataloader):

        # prediction_step ------------------------------------------------------
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        labels = nested_detach(tuple(inputs.get(name) for name in ['labels']))
        if len(labels) == 1:
            labels = labels[0]

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss = loss.mean().detach()
            ignore_keys = []
            if "logits" in outputs:
                logits = outputs["logits"]
            elif isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]
                
            if "hidden_states" in outputs:
                encoder_outputs = outputs["hidden_states"]
                encoder_outputs = nested_detach(encoder_outputs)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if len(logits.size()) == 1: logits = logits.unsqueeze(0)
        losses = loss.repeat(args.batch_size)
        losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
        preds_host = logits if preds_host is None else torch.cat((preds_host, logits), dim=0)
        labels_host = labels if labels_host is None else torch.cat((labels_host, labels), dim=0)
        
        all_encoder_outputs = encoder_outputs if all_encoder_outputs is None else torch.cat((all_encoder_outputs, encoder_outputs), dim=0)
        ### check gradient...?
        
    if losses_host is not None:
        losses = nested_numpify(losses_host)
        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    if preds_host is not None:
        logits = nested_numpify(preds_host)
        all_preds = logits if all_preds is None else np.concatenate((all_preds, logits), axis=0)
    if labels_host is not None:
        labels = nested_numpify(labels_host)
        all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)

    preds = np.argmax(all_preds, axis=1)
    metric = datasets.load_metric("accuracy")
    metrics = metric.compute(predictions=preds, references=all_labels)
    metrics = denumpify_detensorize(metrics)

    metric_key_prefix = "eval"
    if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

    for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)     

    # save predictions ----------------------------------------------------------
    # if preds is not None:
    #     with open("preds_%s.txt"%filename, "a") as f:
    #         for p, l in zip(preds, all_labels):
    #             f.write("%s\t%s\n"%(p,l))

    # if all_encoder_outputs is not None and filename is not None:
    #     print(all_encoder_outputs.size())
    #     pickle.dump(all_encoder_outputs, open(os.path.join(output_dir, filename), "wb"))
    
    return PredictionOutput(predictions=preds, label_ids=all_labels, metrics=metrics)

def save_prediction_for_hans(outputs, filename="hans_preds.txt", output_dir=""):
    preds = outputs.predictions
    with open(os.path.join(output_dir, filename), "w") as writer:
        writer.write('pairID,gold_label\n')
        for i, p in enumerate(preds):
            writer.write(f'ex{i},{p}\n')

def save_outputs(outputs, input_ids, tokenizer, all_values=False, filename="dot_products.pkl", output_dir=""):
    input_ids = input_ids.cpu().numpy()
    input_tokens = []
    for i, ids in tqdm(enumerate(input_ids), desc="Tokens"):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        input_tokens.append(tokens)
        
    input_tokens = np.array(input_tokens)
    print(outputs.shape)
    results = [pair for pair in zip(input_tokens, outputs)]
    pickle.dump(results[:100], open(os.path.join(output_dir, filename), "wb"))

#==============================================================================#
# main function                                                                #
#==============================================================================#
def main(args):

    logging.set_verbosity_error()
    args.device = torch.device("cuda")
    set_random_seed(args.seed)

    # load pretrained model ----------------------------------------------------
    print("Stage 1. Load pretrained model")
    print("model type: %s"%args.model)
    config, tokenizer, model = load_pretrained_model(args)
    model = model.to(args.device)

    # load dataset for training ------------------------------------------------
    print("Stage 2. Load dataset for training")
    label_list = ['entailment', 'neutral', 'contradiction']
    
    if args.do_contrastive:
        contrastive_train_set_path = os.path.join(args.output_dir, args.contrastive_train_set_name) + "_o.tsv"
        print(contrastive_train_set_path)
        o_set = load_dataset_from_file(contrastive_train_set_path, tokenizer=tokenizer, pos_s1=0, pos_s2=1, pos_lb=2, label_list=label_list, file_type='tsv')

        e_df = pd.read_csv(os.path.join(args.output_dir, args.contrastive_train_set_name) + "_e.tsv", delimiter='\t')
        e_texts = e_df[['premise', 'hypothesis']].values.tolist()
        n_df = pd.read_csv(os.path.join(args.output_dir, args.contrastive_train_set_name) + "_n.tsv", delimiter='\t')
        n_texts = n_df[['premise', 'hypothesis']].values.tolist()
        c_df = pd.read_csv(os.path.join(args.output_dir, args.contrastive_train_set_name) + "_c.tsv", delimiter='\t')
        c_texts = c_df[['premise', 'hypothesis']].values.tolist()
        e_set = texts_to_dataset(e_texts, tokenizer=tokenizer) # entailment
        n_set = texts_to_dataset(n_texts, tokenizer=tokenizer) # neutral
        c_set = texts_to_dataset(c_texts, tokenizer=tokenizer) # contradiction   

        contrastive_train_set = combine_datasets([o_set, e_set, n_set, c_set])
    
    # train --------------------------------------------------------------------
    print("Stage 3. Train")
    
    if args.do_contrastive:
        tmp_epochs = args.num_epochs
        tmp_lr = args.learning_rate
        args.learning_rate = args.cl_learning_rate
        args.num_epochs = args.cl_epochs
        
        print('> 1st CL training : ', contrastive_train_set.encodings['input_ids'].size())
        model = train(args, model, contrastive_train_set)
        
        args.learning_rate = tmp_lr
        args.num_epochs = tmp_epochs
    
    train_set = load_dataset_from_file(args.train_set_path, tokenizer=tokenizer, pos_s1=0, pos_s2=1, pos_lb=2, label_list=label_list, file_type='tsv')   
    print('> 2nd Classification training : ', train_set.encodings['input_ids'].size())
    model = train(args, model, train_set)
    
    # save model ---------------------------------------------------------------
    # torch.save(model.state_dict(), f"./models/{args.model}-cfnli-s{args.seed}/  cl-e3-classification-e3.bin")
    
    # load dataset for evaluation ----------------------------------------------
    print("Stage 4. Load dataset for evaluation")
    cf_org_test = load_dataset_from_file('data/cf_snli/original/test.tsv', tokenizer=tokenizer, pos_s1=0, pos_s2=1, pos_lb=-1, label_list=label_list, file_type='tsv')
    cf_rp_test = load_dataset_from_file('data/cf_snli/revised_premise/test.tsv', tokenizer, pos_s1=0, pos_s2=1, pos_lb=-1, label_list=label_list, file_type='tsv')
    cf_rh_test = load_dataset_from_file('data/cf_snli/revised_hypothesis/test.tsv', tokenizer, pos_s1=0, pos_s2=1, pos_lb=-1, label_list=label_list, file_type='tsv')
    snli_test = load_dataset_from_file('data/snli/test.tsv', tokenizer, pos_s1=7, pos_s2=8, pos_lb=-1, label_list=label_list, file_type='tsv')
    mnli_dev_m = load_dataset_from_file('data/mnli/dev_matched.tsv', tokenizer, pos_s1=8, pos_s2=9, pos_lb=-1, label_list=label_list, file_type='tsv')
    mnli_dev_mm = load_dataset_from_file('data/mnli/dev_mismatched.tsv', tokenizer, pos_s1=8, pos_s2=9, pos_lb=-1, label_list=label_list, file_type='tsv')

    # evaluate -----------------------------------------------------------------
    print("Stage 5. Evaluate")
    #print("> Counterfactually Augmented Dataset - test set")
    org_output = evaluate(args, model, cf_org_test, filename="orig")    
    log_metrics("eval", org_output.metrics)
    revp_output = evaluate(args, model, cf_rp_test, filename="rp")    
    log_metrics("eval", revp_output.metrics)
    revh_output = evaluate(args, model, cf_rh_test, filename="rh")    
    log_metrics("eval", revh_output.metrics)
    print("> SNLI Dataset - test set")
    snli_test_output = evaluate(args, model, snli_test)
    log_metrics("eval", snli_test_output.metrics)
    mnli_dev_m_output = evaluate(args, model, mnli_dev_m)
    log_metrics("eval", mnli_dev_m_output.metrics)
    mnli_dev_mm_output = evaluate(args, model, mnli_dev_mm)
    log_metrics("eval", mnli_dev_mm_output.metrics)

    with open("experiment_results_%s.tsv"%args.model, "a") as f:
        if args.do_contrastive: f.write("CONTRASTIVE\t")
        else: f.write("NONE\t")
        if args.train_set_path == "data/cf_snli/original/train.tsv": f.write("ORIGINAL\t")
        else: f.write("AUGMENTED\t")
        f.write("%d\t"%args.seed)
        if args.do_contrastive: f.write("%d\t"%args.cl_epochs)
        else: f.write("0\t")
        if args.do_contrastive: f.write("%f\t"%args.cl_learning_rate)
        else: f.write("0.0\t")
        f.write("%d\t"%args.num_epochs)
        f.write("%f\t"%args.learning_rate)
        f.write("%.2f\t"%(org_output.metrics['eval_accuracy']*100))
        f.write("%.2f\t"%(revp_output.metrics['eval_accuracy']*100))
        f.write("%.2f\t"%(revh_output.metrics['eval_accuracy']*100))
        f.write("%.2f\t"%(snli_test_output.metrics['eval_accuracy']*100))  
        f.write("%.2f\t"%(mnli_dev_m_output.metrics['eval_accuracy']*100))  
        f.write("%.2f\n"%(mnli_dev_mm_output.metrics['eval_accuracy']*100))  

    with open("experiment_results_%s.txt"%args.model, "a") as f:
        f.write("------------------------------------\n")
        f.write(" CONFIGURATION\n")
        f.write("------------------------------------\n")
        if args.do_contrastive: f.write(" 1ST TRAINING: CONTRASTIVE\n")
        else: f.write(" 1ST TRAINING: NONE\n")
        if args.train_set_path == "data/cf_snli/original/train.tsv": f.write(" 2ND TRAINING: ORIGINAL\n")
        else: f.write(" 2ND TRAINING: AUGMENTED\n")
        f.write(" SEED NUMBER : %d\n"%args.seed)
        if args.do_contrastive: f.write(" CLL EPOCHS & LR : %d, %f\n"%(args.cl_epochs, args.cl_learning_rate))
        f.write(" CEL EPOCHS & LR : %d, %f\n"%(args.num_epochs, args.learning_rate))
        f.write("------------------------------------\n")   
        f.write(" EVALUATION RESULTS\n")
        f.write("------------------------------------\n")
        f.write(" CF_SNLI ORIGINIAL         : %.2f%%\n"%(org_output.metrics['eval_accuracy']*100))
        f.write(" CF_SNLI REVISED_PREMISE   : %.2f%%\n"%(revp_output.metrics['eval_accuracy']*100))
        f.write(" CF_SNLI REVISED_HYPOTHESIS: %.2f%%\n"%(revh_output.metrics['eval_accuracy']*100))
        f.write(" SNLI                      : %.2f%%\n"%(snli_test_output.metrics['eval_accuracy']*100))
        f.write(" MNLI-m                    : %.2f%%\n"%(mnli_dev_m_output.metrics['eval_accuracy']*100))
        f.write(" MNLI-mm                   : %.2f%%\n"%(mnli_dev_mm_output.metrics['eval_accuracy']*100))
        f.write("------------------------------------\n")
        f.write("\n")

    print("------------------------------------")
    print(" CONFIGURATION ")
    print("------------------------------------")
    if args.do_contrastive: print(" 1ST TRAINING: CONTRASTIVE")
    else: print(" 1ST TRAINING: NONE")
    if args.train_set_path == "data/cf_snli/original/train.tsv": print(" 2ND TRAINING: ORIGINAL")
    else: print(" 2ND TRAINING: AUGMENTED")
    print(" SEED NUMBER : %d"%args.seed)
    if args.do_contrastive: print(" CLL EPOCHS & LR : %d, %f"%(args.cl_epochs, args.cl_learning_rate))
    print(" CEL EPOCHS & LR : %d, %f"%(args.num_epochs, args.learning_rate))
    print("------------------------------------")   
    print(" EVALUATION RESULTS")
    print("------------------------------------")
    print(" CF_SNLI ORIGINIAL         : %.2f%%"%(org_output.metrics['eval_accuracy']*100))
    print(" CF_SNLI REVISED_PREMISE   : %.2f%%"%(revp_output.metrics['eval_accuracy']*100))
    print(" CF_SNLI REVISED_HYPOTHESIS: %.2f%%"%(revh_output.metrics['eval_accuracy']*100))
    print(" SNLI                      : %.2f%%"%(snli_test_output.metrics['eval_accuracy']*100))
    print(" MNLI-m                    : %.2f%%"%(mnli_dev_m_output.metrics['eval_accuracy']*100))
    print(" MNLI-mm                   : %.2f%%"%(mnli_dev_mm_output.metrics['eval_accuracy']*100))
    print("------------------------------------")
    print(" ")

#==============================================================================#
# entry point                                                                  #
#==============================================================================#
if __name__ == "__main__":
    
    # parse arguments ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1001, type=int, help='random seed')
    parser.add_argument('--model', default='bert-base-uncased', type=str, help='pretrained language model') # roberta-base
    parser.add_argument('--task_name', default='snli', type=str, help='task name')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=3, type=int, help='number of training epochs')
    parser.add_argument('--cl_epochs', default=3, type=int, help='number of training epochs for contrastive learning')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay factor')
    parser.add_argument('--cl_learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    
    parser.add_argument('--do_contrastive', action='store_true', help='do contrastive learning before classification learning')
    parser.add_argument('--contrastive_train_set_name', default='selected_set_i4', type=str, help='train set for contrastive learning')
    parser.add_argument('--train_set_path', default='data/cf_snli/original/train.tsv', type=str, help='train set for classification learning')

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(5678)
        print("waiting for debugger to attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')

    # call main function -------------------------------------------------------    
    os.environ['CUDA_VISIBLE_DEVICES']='0' # gpu
    main(args)
