
data_dir=../data

data=../data/train_data.json
list="char emb synonym wordnet tfidf"

rm ${data_dir}/*_aug.json

for i in $list
do
    cat ${i}_aug.json $data >> ${data_dir}/${i}_aug.json
done

# create rm datasets
python3 process_rm.py --input ${data_dir}/*aug*json