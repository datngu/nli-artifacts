import datasets
import os, sys
import numpy as np
import argparse
#import nlpaug.augmenter.char as nac
#import nlpaug.augmenter.word as naw
#import nlpaug.augmenter.sentence as nas
#import nlpaug.flow as nafc
#from nlpaug.util import Action


parser = argparse.ArgumentParser(description = "Remove premise NLI datasets...)",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input", nargs='+', required = True, help = "json data input - multiple files accepted - file need to be reable by hugging face dataset method")
#parser.add_argument('--out', type=str, default = 'out.json', help= "output argumentated data")


def remove_premise(data):
    new_data = []
    for ex in data:
        #premise = ex['premise']
        premise = ''
        hypothesis = ex['hypothesis']
        label = ex['label']
        if label == -1: continue
        tem_dict = {'premise': premise, 'hypothesis': hypothesis, 'label': label}
        new_data.append(tem_dict)
    return new_data


def write_json(data, fn):
    import json
    with open(fn, 'w') as json_file:
        for d in data:
            if d['label'] > -1:
                json.dump(d, json_file)
                json_file.write('\n')



def process_data(data_path):
    data_path = os.path.abspath(data_path)
    print(f'processing...{data_path}')
    data = datasets.load_dataset('json', data_files = data_path)['train']
    new_data = remove_premise(data)
    #new_path = os.path.dirname(data_path) + '/rm_premise_' + os.path.basename(data_path)
    new_path = data_path.replace('.json', '_rm.json')
    print(f'writing...{new_path}')
    write_json(new_data, new_path)


if __name__ == '__main__':
    args = parser.parse_args()
    input = args.input
    for fi in input:
        process_data(fi)




