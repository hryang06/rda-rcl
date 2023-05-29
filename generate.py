import os
from tqdm import tqdm
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

import spacy
# import nltk
# from nltk.corpus import wordnet
# from nltk.stem.wordnet import WordNetLemmatizer
from pattern.en import wordnet, pluralize
from pattern.en import NOUN, VERB, ADJ

nlp = spacy.load('en_core_web_sm')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # random.seed(seed)
    np.random.seed(seed)

#==============================================================================#
# TokenFrequency                                                               #
#==============================================================================#
class TokenFrequency():
    def __init__(self, dataset, is_lower=True):
        self.is_lower = is_lower
        self.words = []
        self.frequency = self.get_word_frequency(dataset)
        
    def get_word_frequency(self, dataset):
        for text in dataset.texts:
            doc = nlp(text[0].lower()) if self.is_lower else nlp(text[0])
            self.words += [tok.text for tok in doc]
            doc = nlp(text[1].lower()) if self.is_lower else nlp(text[1])
            self.words += [tok.text for tok in doc]

        return Counter(self.words)
    
    def get_frequency(self, wordlist, return_tensors='pt'):
        freq_list = [self.frequency[w.lower() if self.is_lower else w] for w in wordlist]
        if return_tensors == 'pt':
            freq_list = torch.tensor(freq_list)
        elif return_tensors == 'np':
            freq_list = np.array(freq_list)
        return freq_list

    def get_wf_pair(self, wordlist, is_sorted=True): # word - frequency pair
        if len(self.words) == 0 or len(wordlist) == 0: return []
        
        pairs = [(w, self.frequency[w.lower() if self.is_lower else w]) for w in wordlist]
        if is_sorted:
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        return pairs
    
    def get_frequency_softmax(self, wordlist=None, freq=None):
        if len(self.words) == 0: return []
        if wordlist is None and freq is None:
            raise NotImplementedError
        
        if freq is None:
            freq = self.get_frequency(wordlist).float()
        freq_softmax = F.softmax(freq, dim=0)
        return freq_softmax

    def find_most_frequent_words(self, strlist):
        word_list = []
        for w in strlist:                        
            if len(w.split('-')) > 1:  w = ' '.join(w.split('-'))
            if len(w.split('_')) > 1:  w = ' '.join(w.split('_'))
            
            word_list.append(w)
            
        # sort by frequency
        most_word = (None, 0)
        if len(word_list) > 0:
            freq_pair = self.get_wf_pair(word_list)[0]
            # freq = self.get_frequency(word_list).float()
            # freq_p = self.get_frequency_softmax(freq=freq)
            # idx = torch.multinomial(freq_p, 1)[0]
            # most_word = (word_list[idx], freq[idx])
            
            if freq_pair[1] > 0:
                most_word = freq_pair
            else:
                # random
                # 만약 빈도수가 동일한 경우에도 랜덤으로 선택해야할까...?
                # 그리고 빈도수가 0인 경우에는 그냥 DA를 안하는게 나을까?
                idx = np.random.randint(len(word_list))
                most_word = (word_list[idx], 0)
        return most_word

#==============================================================================#
# class TokenPOS                                                               #
#==============================================================================#
class TokenPOS():
    def __init__(self, doc):
        self.doc = doc
        
        self.all_noun = []
        self.main_noun = []
        self.sub_noun = []
        self.all_adj = []
        
        self.seperate_by_pos()
    
    def seperate_by_pos(self):
        for k, tok in enumerate(self.doc):
            if tok.pos_ == 'NOUN':
                if tok.dep_ in ['nsubj', 'ROOT']: # main noun
                    self.main_noun.append([k, tok])
                else:
                    self.sub_noun.append([k, tok])
            elif tok.pos_ == 'ADJ':
                self.all_adj.append([k, tok])
            # elif tok.tag_ in ['VB', 'VBP']:
            #     base_verb.append([k, tok])
        self.all_noun = self.main_noun + self.sub_noun


def replace_word_in_tokens(doc, target, replace_word, return_type='str'):
    tokens = [[tok.text, tok.whitespace_] for tok in doc]

    if isinstance(target, int):
        tok = doc[target]
        # if tok.tag_ in ['NNS', 'NNPS']: replace_word = pluralize(replace_word)
        
        # if tok.shape_.isupper(): replace_word.upper()
        # elif tok.shape_[:1].isupper():
        #     replace_word = replace_word[:1].upper() + replace_word[1:]
        
        tokens[target][0] = replace_word
    else:
        raise NotImplementedError
    
    if return_type == 'str': # to string text
        texts = ''
        for tok in tokens:
            texts += tok[0] + tok[1]
        return texts
    else:
        return tokens

def unify_to_strlist(words, ignore_word=None):
    all_words = []
    for word in words:
        if isinstance(word, wordnet.Synset):
            # wordlist = word.synonyms # add all synonyms
            wordlist = [word.synonyms[0]]
        elif isinstance(word, str):
            wordlist = [word]
        else:
            raise NotImplementedError

        for w in wordlist:
            # if w in all_words: continue # no repeat
            if isinstance(ignore_word, str) and w == ignore_word: continue
            elif isinstance(ignore_word, list) and w in ignore_word: continue
            all_words.append(w)
            
    return all_words

# def find_appropriate_synset(lemma, pos=NOUN):
#     syns = wordnet.synsets(lemma, pos=pos)
    
#     syn = None
#     for syn in syns:
#         if lemma in syn.synonyms: break
    
#     return syn

def revise_to_entailment(targets, frequency, reviseTo='premise'):
    # ENTAILMENT: replace to synonym or lexname/hyponym
    e_word, e_idx, max_freq = None, None, -1
    for idx, tok in targets:
        syns = wordnet.synsets(tok.lemma_, pos=NOUN)
        if len(syns) == 0: continue
        target_syn = syns[0]
        
        syn_syns = target_syn.synonyms # str list
        if reviseTo == 'hypothesis':
            lex_word = target_syn.lexname.split('.')[1:]
            if 'Tops' in lex_word: lex_word.remove('Tops')
            # hyper_syns = target_syn.hypernyms(recursive=True, depth=3)
            all_syns = syn_syns + lex_word# + hyper_syns
        elif reviseTo == 'premise': # hypothesis
            hypo_syns = target_syn.hyponyms(recursive=True, depth=3)
            all_syns = syn_syns + hypo_syns
        else:
            raise NotImplementedError
        
        all_words = unify_to_strlist(all_syns, ignore_word=tok.lemma_)
        word, freq = frequency.find_most_frequent_words(all_words)
        if word is not None and freq > max_freq:
            e_word = word
            e_idx = idx
            max_freq = freq
        
    return e_word, e_idx

def revise_to_neutral(targets, frequency, reviseTo='premise'):
    # NEUTRAL: replace to hyponym/lexname
    n_word, n_idx, max_freq = None, None, -1
    for idx, tok in targets:
        syns = wordnet.synsets(tok.lemma_, pos=NOUN)
        if len(syns) == 0: continue
        target_syn = syns[0]
        ignore_words = target_syn.synonyms
        
        if reviseTo == 'hypothesis':
            all_syns = target_syn.hyponyms(recursive=True, depth=3)
        elif reviseTo == 'premise':
            lex_word = target_syn.lexname.split('.')[1:]
            if 'Tops' in lex_word: lex_word.remove('Tops')
            # hyper_syns = target_syn.hypernyms(recursive=True, depth=3)
            all_syns = lex_word# + hyper_syns
        else:
            raise NotImplementedError
        
        all_words = unify_to_strlist(all_syns, ignore_word=ignore_words)
        word, freq = frequency.find_most_frequent_words(all_words)
        if word is not None and freq > max_freq:
            n_word = word
            n_idx = idx
            max_freq = freq
            
    return n_word, n_idx

def revise_to_contradiction(targets, frequency, pos=NOUN):
    # CONTRADICTION: replace to antonym
    c_word, c_idx, max_freq = None, None, -1
    for idx, tok in targets:
        syns = wordnet.synsets(tok.lemma_, pos=pos)
        if len(syns) == 0: continue
        target_syn = syns[0]
        ignore_words = target_syn.synonyms
        
        anto_syns = target_syn.antonym
        anto_syns = anto_syns if anto_syns is not None else []
        
        hyper_hypo, hypo_hyper = [], []
        if len(target_syn.hypernyms()) > 0:
            hyper_hypo = target_syn.hypernyms()[0].hyponyms()
        if len(target_syn.hyponyms()) > 0:
            hypo_hyper = target_syn.hyponyms()[0].hypernyms()
        all_syns = anto_syns + hyper_hypo + hypo_hyper
        
        all_words = unify_to_strlist(all_syns, ignore_word=ignore_words)
        word, freq = frequency.find_most_frequent_words(all_words)
        if word is not None and freq > max_freq:
            c_word = word
            c_idx = idx
            max_freq = freq
    
    return c_word, c_idx

#==============================================================================#
# WordNet Data Augmentation by Copying                                         #
#==============================================================================#
def wda_by_copying(dataset, frequency=None, copy_type='premise', revise_type='hypothesis', is_entailment=True, is_neutral=True, is_contradiction=True, num_samples=None, use_mask=False):
    if copy_type not in ['premise', 'hypothesis'] or revise_type not in ['premise', 'hypothesis']:
        raise NotImplementedError
    
    text_idx = 0 if copy_type == 'premise' else 1
    if frequency is None:
        frequency = TokenFrequency(dataset)
        
    cnt = 0
    mask_tok = '<mask>' #'[MASK]' # bert
    num_data = len(dataset.labels)
    da_mask = np.zeros((num_data, 3))
    new_set = [[], [], []]
    for i, texts in enumerate(tqdm(dataset.texts)):
        if isinstance(num_samples, int) and i >= num_samples: break
        
        # if i not in [141,180,196,199,320,398,672,704,734,737,1037,1041,1199,1205,1241,1418,1429,1505,1530,1598,1599]: continue
        # print('>>', i, texts[0])
        
        text = texts[text_idx] # use only premise sentence
        doc = nlp(text)

        # find nsubj or Root noun
        pos_info = TokenPOS(doc)
        all_noun = pos_info.all_noun
        
        flag = 0
        if is_entailment:
            e_word, idx = revise_to_entailment(all_noun, frequency, reviseTo=revise_type)
            if e_word is not None:
                da_mask[i, 0] = 1
                new_text = replace_word_in_tokens(doc, idx, e_word)
                output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
                new_set[0].append(output)
            elif use_mask and len(all_noun) > 0:
                idx = all_noun[0][0]
                new_text = replace_word_in_tokens(doc, idx, mask_tok)
                output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
                new_set[0].append(output)
                flag += 1
            else:
                new_set[0].append(texts) # add original texts
        
        if is_neutral:
            n_word, idx = revise_to_neutral(all_noun, frequency, reviseTo=revise_type)
            if n_word is not None:
                da_mask[i, 1] = 1
                new_text = replace_word_in_tokens(doc, idx, n_word)
                output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
                new_set[1].append(output)
            elif use_mask and len(all_noun) > 0:
                idx = all_noun[0][0]
                new_text = replace_word_in_tokens(doc, idx, mask_tok)
                output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
                new_set[1].append(output)
                flag += 1
            else:
                new_set[1].append(texts) # add original texts
        
        if is_contradiction:
            c_word, idx = revise_to_contradiction(all_noun, frequency)
            # if c_word is None:
            #     c_word, idx = revise_to_contradiction(all_adj, frequency, pos=ADJ)
            if c_word is not None:
                da_mask[i, 2] = 1
                new_text = replace_word_in_tokens(doc, idx, c_word)
                output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
                new_set[2].append(output)
            elif use_mask and len(all_noun) > 0:
                idx = all_noun[0][0]
                new_text = replace_word_in_tokens(doc, idx, mask_tok)
                output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
                new_set[2].append(output)
                flag += 1
            else:
                new_set[2].append(texts) # add original texts
            
            if flag > 1: cnt += 1
        # cnt += 1
        # if cnt > 100: break

    print(f'mask cnt : {cnt}')
    print(f"> WDA Copy Premise: {np.sum(da_mask)}")
    print(f"> entailment:\t {sum(da_mask[:,0])}/{num_data}")
    print(f"> neutral:\t {sum(da_mask[:,1])}/{num_data}")
    print(f"> contradiction: {sum(da_mask[:,2])}/{num_data}")

    return new_set

#==============================================================================#
# WordNet Data Augmentation by Copying                                         #
#==============================================================================#
def wda_by_copying_with_drop(dataset, frequency=None, copy_type='premise', revise_type='hypothesis', num_samples=None):
    if copy_type not in ['premise', 'hypothesis'] or revise_type not in ['premise', 'hypothesis']:
        raise NotImplementedError
    
    text_idx = 0 if copy_type == 'premise' else 1
    if frequency is None:
        frequency = TokenFrequency(dataset)
    
    label_list = ["entailment", "neutral", "contradiction"]
    labels = [label_list[l] for l in dataset.labels]
        
    cnt = 0
    num_data = len(labels)
    da_mask = np.zeros((num_data, 3))
    all_texts, all_labels = [[],[]], []
    org_texts, org_labels = [[],[]], []
    e_texts, n_texts, c_texts = [[],[]], [[],[]], [[],[]]
    for i, texts in enumerate(tqdm(dataset.texts)):
        if isinstance(num_samples, int) and i >= num_samples: break
        
        text = texts[text_idx] # use only premise sentence
        doc = nlp(text)

        # find nsubj or Root noun
        pos_info = TokenPOS(doc)
        all_noun = pos_info.all_noun
        
        all_pos = True
        outputs = []
        
        all_texts[0].append(texts[0])
        all_texts[1].append(texts[1])
        all_labels.append(labels[i])
        
        e_word, idx = revise_to_entailment(all_noun, frequency, reviseTo=revise_type)
        if e_word is not None:
            da_mask[i, 0] = 1
            new_text = replace_word_in_tokens(doc, idx, e_word)
            output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
            outputs.append(output)
            all_texts[0].append(output[0])
            all_texts[1].append(output[1])
            all_labels.append("entailment")
        else: all_pos = False

        if all_pos:
            n_word, idx = revise_to_neutral(all_noun, frequency, reviseTo=revise_type)
            if n_word is not None:
                da_mask[i, 1] = 1
                new_text = replace_word_in_tokens(doc, idx, n_word)
                output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
                outputs.append(output)
                
                all_texts[0].append(output[0])
                all_texts[1].append(output[1])
                all_labels.append("neutral")
            else: all_pos = False
    
        if all_pos:
            c_word, idx = revise_to_contradiction(all_noun, frequency)
            # if c_word is None:
            #     c_word, idx = revise_to_contradiction(all_adj, frequency, pos=ADJ)
            if c_word is not None:
                da_mask[i, 2] = 1
                new_text = replace_word_in_tokens(doc, idx, c_word)
                output = [text, new_text] if revise_type == 'hypothesis' else [new_text, text]
                outputs.append(output)
                
                all_texts[0].append(output[0])
                all_texts[1].append(output[1])
                all_labels.append("contradiction")
            else: all_pos = False
        
        if all_pos:
            e_texts[0].append(outputs[0][0])
            e_texts[1].append(outputs[0][1])
            
            n_texts[0].append(outputs[1][0])
            n_texts[1].append(outputs[1][1])
            
            c_texts[0].append(outputs[2][0])
            c_texts[1].append(outputs[2][1])
            
            org_texts[0].append(dataset.texts[i][0])
            org_texts[1].append(dataset.texts[i][1])
            org_labels.append(labels[i])
        # cnt += 1
        # if cnt > 100: break
    
    all_set = all_texts + [all_labels]
    org_set = org_texts + [org_labels]
    
    print(f'mask cnt : {cnt}')
    print(f"> WDA Copy Premise: {np.sum(da_mask)}")
    print(f"> entailment:\t {sum(da_mask[:,0])}/{num_data}")
    print(f"> neutral:\t {sum(da_mask[:,1])}/{num_data}")
    print(f"> contradiction: {sum(da_mask[:,2])}/{num_data}")

    return all_set, org_set, e_texts, n_texts, c_texts

def save_da_to_tsv(texts, label_list=None, filename="wordnet_da_set.tsv", output_dir=""):
    with open(os.path.join(output_dir, filename), 'w') as writer:
        writer.write(f'sentence1\tsentence2\tlabel\n')
        for i, c in enumerate(texts):
            for t in c:
                l = i
                if label_list is not None:
                    l = label_list[i]
                writer.write(f'{t[0]}\t{t[1]}\t{l}\n')
