#!/bin/bash

cp data/cf_snli/original/train.tsv outputs/selected_set_i0.tsv

for s in 1001 1002 1003 1004 1005
do
    python3 augment.py --original_set_path=data/cf_snli/original/train.tsv --seed $s --selected_set_name=token_based_data
done

# for c in premise hypothesis
# do
#     for r in premise hypothesis
#     do
#         # filter bad data and make the next training set
#         python3 augment.py --original_set_path=data/cf_snli/original/train.tsv --selected_set_name=C$c-R$r --copy_text=$c --revise_text=$r
#     done
# done
