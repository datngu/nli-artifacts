#!/bin/bash        
#SBATCH --job-name=char
#SBATCH --output=char.out  
#SBATCH --nodes=1    
#SBATCH --mem=64G
#SBATCH --ntasks=16               
#SBATCH --partition=gpu
#SBATCH --mail-user=n.dat@outlook.com
#SBATCH --mail-type=ALL


process_aug() {
    data=$1
    task=$2
    model=$3
    number_of_parts=10
    total_lines=$(wc -l < "$data")
    lines_per_part=$((total_lines / number_of_parts))
    split -l $lines_per_part $data partion_${task}
    for fi in partion_${task}*
    do
        python do_augmentation.py --input $fi --out ${fi}.json --task $task --model $model &
    done
    wait 
    cat partion_${task}*.json >> ${task}_aug.json
    rm partion_${task}*
}

data=../data/train_data.json
model=nothing

# char
task='char'
process_aug $data $task $model

# tfidf
task='tfidf'
process_aug $data $task $model


# wordnet
task='wordnet'
process_aug $data $task $model


# synonym
task='synonym'
model=/mnt/SCRATCH/ngda/nlp/data/aug_model/ppdb-2.0-m-all
process_aug $data $task $model


# emb
task='emb'
model=/mnt/SCRATCH/ngda/nlp/data/aug_model/GoogleNews-vectors-negative300.bin
process_aug $data $task $model

