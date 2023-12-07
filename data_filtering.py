import os
import random
import torch.nn.functional as F
import pytorch_lightning as pl
pl.seed_everything(42)

from tqdm import tqdm
from utils import load_json, save_json
from utils.triplet import parse_triplets_seq_char, make_triplets_seq, parse_triplets_seq
from collections import defaultdict, Counter


def load_origin(origin_data_dir):
    train_file_name = os.path.join(origin_data_dir, 'train.json')
    dev_file_name   = os.path.join(origin_data_dir, 'dev.json')
    test_file_name  = os.path.join(origin_data_dir, 'test.json')

    datasets = {
        'train': load_json(train_file_name),
        'dev'  : load_json(dev_file_name),
        'test' : load_json(test_file_name),
    }

    for example in datasets['train']:
        example['triplets_seq'] = make_triplets_seq(example['triplets'])

    return datasets



def save_data(output_examples, datasets, output_dir):
    output_dir1 = os.path.join(output_dir, 'augmented')
    output_dir2 = os.path.join(output_dir, 'augmented_origin')

    save_json(output_examples,  os.path.join(output_dir1, 'train.json'))
    save_json(datasets['dev'],  os.path.join(output_dir1, 'dev.json'))
    save_json(datasets['test'], os.path.join(output_dir1, 'test.json'))

    save_json(output_examples+datasets['train'], os.path.join(output_dir2, 'train.json'))
    save_json(datasets['dev'],  os.path.join(output_dir2, 'dev.json'))
    save_json(datasets['test'], os.path.join(output_dir2, 'test.json'))



def mean(lst):
    return sum(lst) / len(lst)



def random_round(num):
    n = int(num)
    n = n + int(random.random()<(num-n))
    return n



def alignment_selection(examples, num, start):
    if start + num > len(examples):
        end_flag = True
    else:
        end_flag = False

    examples_by_triplet_num = defaultdict(list)
    for example in examples:
        triplet_num = example['triplets_seq'].count(' ; ') + 1
        examples_by_triplet_num[triplet_num].append(example)

    total_num = len(examples)
    output_examples = []
    for triplet_num in sorted(examples_by_triplet_num.keys()):
        bin_num = len(examples_by_triplet_num[triplet_num])
        
        start_num = bin_num / total_num * start
        start_num = round(start_num)

        sample_num = bin_num / total_num * num
        sample_num = random_round(sample_num)
        
        print(triplet_num, bin_num, sample_num)

        if end_flag:
            sampled_examples = sorted(examples_by_triplet_num[triplet_num], key=lambda it: it['alignment_score'], reverse=True)[start_num:]
        else:
            sampled_examples = sorted(examples_by_triplet_num[triplet_num], key=lambda it: it['alignment_score'], reverse=True)[start_num:start_num+sample_num]

        output_examples.extend(sampled_examples)

    return output_examples



def fluency_selection(examples, start, num, bin_width=4):
    if start + num == len(examples):
        end_flag = True
    else:
        end_flag = False

    examples_by_length = defaultdict(list)
    for example in examples:
        length = example['sentence'].count(' ') + 1
        examples_by_length[length//bin_width].append(example)

    total_num = len(examples)
    output_examples = []
    for length in sorted(examples_by_length.keys()):
        bin_num = len(examples_by_length[length])
        
        start_num = bin_num / total_num * start
        start_num = round(start_num)

        sample_num = bin_num / total_num * num
        sample_num = random_round(sample_num)
        
        print(length, bin_num, sample_num)

        if end_flag:
            sampled_examples = sorted(examples_by_length[length], key=lambda it: it['fluency_score'], reverse=True)[start_num:]
        else:
            sampled_examples = sorted(examples_by_length[length], key=lambda it: it['fluency_score'], reverse=True)[start_num:start_num+sample_num]

        output_examples.extend(sampled_examples)

    return output_examples


def stat_augmentation(examples):
    total_num = len(examples)
    aligned_examples = [example for example in examples if example['alignment_score']>0]
    fluency_examples = [example for example in examples if example['fluency_score']>0]
    
    aligned_scores = [example['alignment_score'] for example in examples]
    fluency_scores = [example['fluency_score'] for example in examples]

    print(f'aligned: {len(aligned_examples)}/{total_num} | avg: {mean(aligned_scores):.2f}, min: {min(aligned_scores):.2f}')
    print(f'fluency: {len(fluency_examples)}/{total_num} | avg: {mean(fluency_scores):.2f}, min: {min(fluency_scores):.2f}')


def cal_sim(example1, example2):
    triplet1 = example1['triplets_parsed']
    triplet2 = example2['triplets_parsed']

    aspect_overlap  = sum([t1[0].lower()==t2[0].lower() for t1, t2 in zip(triplet1, triplet2)])
    opinion_overlap = sum([t1[1].lower()==t2[1].lower() for t1, t2 in zip(triplet1, triplet2)])

    overlap = (aspect_overlap+opinion_overlap)

    return overlap / (len(triplet1) + len(triplet2))


def main(args):
    augmented_examples = [
        {
            'ID': example['ID'],
            'triplets_seq': example["triplets_seqq"],
            # 'syn_seq' : example["syn_seq"],
            'sentence': example['generated_sentence'],
            'alignment_score': example['alignment_score'],
            'fluency_score'  : example['fluency_score'],
        }
        for example in load_json(args.augmented_data_dir)
    ]
    datasets = load_origin(args.origin_data_dir)

    stat_augmentation(augmented_examples)

    remove_examples1 = alignment_selection(augmented_examples, start=20_000, num=80_000)
    remove_examples2 = [example for example in augmented_examples if example['fluency_score']<0]
    remove_examples3 = [example for example in augmented_examples if example['alignment_score']<0]

    remove_IDs = [example['ID'] for example in remove_examples1] + [example['ID'] for example in remove_examples2] + [example['ID'] for example in remove_examples3]
    remove_IDs = set(remove_IDs)

    augmented_examples = [example for example in augmented_examples if example['ID'] not in remove_IDs]
    
    print('augmented_examples', len(augmented_examples))

    augmented_examples = random.sample(augmented_examples, k=args.k)

    stat_augmentation(augmented_examples)

    print(f'save {len(augmented_examples)} to {args.output_dir}')
    save_data(augmented_examples, datasets, args.output_dir)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--origin_data_dir', type=str, default='data/origin/14res')
    parser.add_argument('--augmented_data_dir', type=str, default='./output/augmentation/14res_100k_42.json')
    parser.add_argument('--output_dir', type=str, default='./output/augmentation_filtered/14res_5k_42')


    parser.add_argument('--k',  type=int, default=5000)
    parser.add_argument('--seed',  type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    main(args)

"""
python data_filtering.py --origin_data_dir data/origin/14res --augmented_data_dir ./output/augmentation/14res_100k_42.json --output_dir ./output/augmentation_filtered/14res_5k_42 --k 5000
"""