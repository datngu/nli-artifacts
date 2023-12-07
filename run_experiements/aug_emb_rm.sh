#!/bin/bash            
#SBATCH --job-name=emb_aug_rm
#SBATCH --output=emb_aug_rm.out  
#SBATCH --nodes=1    
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16               
#SBATCH --partition=gpu
#SBATCH --mail-user=n.dat@outlook.com
#SBATCH --mail-type=ALL



tag='emb_aug_rm'
eval_tag='rm'


data=/mnt/SCRATCH/ngda/nlp/fp/data/${tag}.json
val_data=/mnt/SCRATCH/ngda/nlp/fp/data/val_${eval_tag}.json
test_data=/mnt/SCRATCH/ngda/nlp/fp/data/test_${eval_tag}.json


#val_data="val_no_premise.json"

out_model=${tag}_model
out_validation=${tag}_eval.txt


python3 ../run/run.py \
    --do_train \
    --task nli \
    --dataset ${data} \
    --per_device_train_batch_size 256 \
    --output_dir ./${out_model}/


python3 ../run/nli_eval.py \
    --val_data ${val_data} \
    --test_data ${test_data} \
    --model_path ./${out_model} \
    --out ${out_validation}