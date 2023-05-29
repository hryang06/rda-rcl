#!/bin/bash

model_name="bert-base-uncased"
learning_rate=3e-5
cl_learning_rate=1e-5
epochs=5
cl_epochs=10

#model_name="roberta-base"
#learning_rate=1e-5
#cl_learning_rate=2e-6
#epochs=5
#cl_epochs=10

da_type=sentence_based_roberta #token_based_data

# weight=1.0
# s=1001

for s in 1001 1003 1004 1002 1005
do
    # 1. 1st: NONE, 2nd: ORIGINAL
    # python3 train_cl.py --seed=$s --train_set_path="data/cf_snli/original/train.tsv" --model=$model_name --learning_rate=$learning_rate --cl_learning_rate=$cl_learning_rate --num_epochs=$epochs

    # SUOERVISED CL
    # python3 train_cl.py --seed=$s --do_contrastive --contrastive_train_set_name="cfsnli_original_train_data" --train_set_path="data/cf_snli/original/train.tsv" --model=$model_name --learning_rate=$learning_rate --cl_learning_rate=$cl_learning_rate --num_epochs=$epochs --cl_epochs=$cl_epochs

    # 3. 1st: NONE, 2nd: AUGMENTED
    # python3 train_cl.py --seed=$s --train_set_path=outputs/$da_type.tsv --model=$model_name --learning_rate=$learning_rate --cl_learning_rate=$cl_learning_rate --num_epochs=$epochs

    # 2. 1st: CONTRASTIVE, 2nd: ORIGINAL
    # python3 train_cl.py --seed=$s --do_contrastive --contrastive_train_set_name=$da_type --train_set_path="data/cf_snli/original/train.tsv" --model=$model_name --learning_rate=$learning_rate --cl_learning_rate=$cl_learning_rate --num_epochs=$epochs --cl_epochs=$cl_epoch
    
    # 4. 1st: CONTRASTIVE, 2nd: AUGMENTED
    #da_type=all_based_roberta
    #python3 train_cl.py --seed=$s --do_contrastive --contrastive_train_set_name=$da_type --train_set_path=outputs/$da_type.tsv --model=$model_name --learning_rate=$learning_rate --cl_learning_rate=$cl_learning_rate --num_epochs=$epochs --cl_epochs=$cl_epochs

    da_type=sentence_based_data
    python3 train_cl.py --seed=$s --do_contrastive --contrastive_train_set_name=$da_type --train_set_path=outputs/$da_type.tsv --model=$model_name --learning_rate=$learning_rate --cl_learning_rate=$cl_learning_rate --num_epochs=$epochs --cl_epochs=$cl_epochs
    
    da_type=token_based_data_s$s
    python3 train_cl.py --seed=$s --do_contrastive --contrastive_train_set_name=$da_type --train_set_path=outputs/$da_type.tsv --model=$model_name --learning_rate=$learning_rate --cl_learning_rate=$cl_learning_rate --num_epochs=$epochs --cl_epochs=$cl_epochs

    # 5. WEIGHTED SUM (CONTRASTIVE & CLASSIFICATION)
    # python3 train_weighted_sum.py --seed=$s --do_contrastive --contrastive_train_set_name=$da_type --train_set_path=outputs/$da_type.tsv --model=$model_name --learning_rate=$learning_rate --num_epochs=$epochs --weight=$weight
done

