import debugpy
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich.console import Console

from util import load_dataset_from_tsvfile
from generate import TokenFrequency, wda_by_copying_with_drop

def set_random_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # random.seed(seed)
    np.random.seed(seed)

def count_samples(df):
    e = len(df[df['label'] == 'entailment'])
    n = len(df[df['label'] == 'neutral'])
    c = len(df[df['label'] == 'contradiction'])
    return e, n, c

def main(args):
    all_set, original_set = [[], [], []], [[], [], []]
    entailment_set, neutral_set, contradiction_set = [[], []], [[], []], [[], []]
    
    set_random_seed(args.seed)
    print("\n*****  SEED %d  *****\n"%args.seed)
    
    # load dataset for training ------------------------------------------------
    print("[LOADING DATASET...]")
    label_list = ['entailment', 'neutral', 'contradiction']
    cf_org_train = load_dataset_from_tsvfile(args.original_set_path, tokenizer=None, pos_s1=0, pos_s2=1, pos_lb=-1, label_list=label_list)
    
    # get frequency ------------------------------------------------------------
    print("[GETTING FREQUENCY...]")
    frequency = TokenFrequency(cf_org_train)
    
    labels = [label_list[l] for l in cf_org_train.labels]
    for i, texts in enumerate(cf_org_train.texts):
        all_set[0].append(texts[0])
        all_set[1].append(texts[1])
        all_set[2].append(labels[i])
            
    for copy_text in ["premise", "hypothesis"]:
        for revise_text in ["premise", "hypothesis"]:
            print("\n------------------------------------")
            print("COPY : %s"%copy_text)
            print("REVISE : %s"%revise_text)
            print("NAME : %s"%args.selected_set_name)
            print()

            # load dataset for training ------------------------------------------------
            print("[LOADING DATASET...]")
            label_list = ['entailment', 'neutral', 'contradiction']
            cf_org_train = load_dataset_from_tsvfile(args.original_set_path, tokenizer=None, pos_s1=0, pos_s2=1, pos_lb=-1, label_list=label_list)
            
            # augment token-level ------------------------------------------------------
            print("[AUGMENTING TOKEN-LEVEL...]")
            _, org_data, e_texts, n_texts, c_texts = wda_by_copying_with_drop(cf_org_train, frequency=frequency, copy_type=copy_text, revise_type=revise_text)
                        
            # concatenate
            for i in range(3):
                original_set[i] += org_data[i]
                if i == 2: break
                
                entailment_set[i] += e_texts[i]
                neutral_set[i] += n_texts[i]
                contradiction_set[i] += c_texts[i]
    
    num_ex = len(original_set[0])
    for i in range(2):
        all_set[i] = all_set[i] + entailment_set[i] + neutral_set[i] + contradiction_set[i]
    all_set[2] = all_set[2] + ['entailment']*num_ex + ['neutral']*num_ex + ['contradiction']*num_ex
    
    # save augmented data ------------------------------------------------------
    print("[SAVING DATA...]")
    save_path = os.path.join(args.output_dir, args.selected_set_name)
    print("save path: %s"%save_path)
    
    all_df = pd.DataFrame({'premise' : all_set[0], 'hypothesis' : all_set[1], 'label' : all_set[2]})
    org_df = pd.DataFrame({'premise' : original_set[0], 'hypothesis' : original_set[1], 'label' : original_set[2]})
    e_df = pd.DataFrame({'premise' : entailment_set[0], 'hypothesis' : entailment_set[1], 'label' : ['entailment']*num_ex})
    n_df = pd.DataFrame({'premise' : neutral_set[0], 'hypothesis' : neutral_set[1], 'label' : ['neutral']*num_ex})
    c_df = pd.DataFrame({'premise' : contradiction_set[0], 'hypothesis' : contradiction_set[1], 'label' : ['contradiction']*num_ex})

    all_df.to_csv(save_path+"_s%d.tsv"%args.seed, sep="\t", index=False)
    org_df.to_csv(save_path+"_s%d_o.tsv"%args.seed, sep="\t", index=False)
    e_df.to_csv(save_path+"_s%d_e.tsv"%args.seed, sep="\t", index=False)
    n_df.to_csv(save_path+"_s%d_n.tsv"%args.seed, sep="\t", index=False)
    c_df.to_csv(save_path+"_s%d_c.tsv"%args.seed, sep="\t", index=False)

    print("total number of sentence-pair sets: %d"%len(org_df))


# program entry point ----------------------------------------------------------
if __name__ == "__main__":

    # parse arguments ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='turn on debug mode')
    parser.add_argument('--output_dir', default="outputs", type=str, help='directory for saving output')
    parser.add_argument('--seed', default=1001, type=int, help='random seed number')

    parser.add_argument('--original_set_path', default="data/cf_snli/original/train.tsv", type=str, help='original data path')
    
    parser.add_argument('--selected_set_name', default="selected_set_i0", type=str, help='filename for selected set')
    parser.add_argument('--copy_text', default='premise', type=str, help='copy text')
    parser.add_argument('--revise_text', default='premise', type=str, help='revise text')
    
    args = parser.parse_args()

    if args.debug is True:
        debugpy.listen(5678)
        print("waiting for debugger to attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')

    args.console = Console(record=True)

    main(args)
