import os
import spacy
import ujson as json
from tqdm import tqdm

from utils import load_json, save_json
from utils import tgenerate_batch



def syn2seq(doc):
    root = _get_root(doc)
    return _syn2seq(root)



def _get_root(doc):
    token = doc[0]
    while True:
        token = token.head
        if token.dep_ == 'ROOT':
            return token



def _syn2seq(token):
    children = list(token.children)
    if len(children) > 0:
        children_seq = ' [' + ' '.join([_syn2seq(child) for child in children]) + ']'
    else:
        children_seq = ''
    return token.dep_ + children_seq



def _parsing(nlp, texts):
    syn_seqs = []
    for doc in nlp.pipe(texts, disable=['ner'], n_process=1):
        seq = syn2seq(doc)
        syn_seqs.append(seq)

    return syn_seqs



def parsing(nlp, dataset):
    for batch_examples in tgenerate_batch(dataset, args.batch_size):
        texts = [example['sentence'] for example in batch_examples]
        for example, syn_seq in zip(batch_examples, _parsing(nlp, texts)):
            example['syn_seq'] = syn_seq

    return dataset



def main(args):
    nlp = spacy.load(args.spacy_model)

    train_file_name = os.path.join(args.data_dir, 'train.json')
    dev_file_name   = os.path.join(args.data_dir, 'dev.json')
    test_file_name  = os.path.join(args.data_dir, 'test.json')

    datasets = {
        'train': load_json(train_file_name),
        'dev'  : load_json(dev_file_name),
        'test' : load_json(test_file_name),
    }

    for mode in ('train', 'dev', 'test'):
        dataset = datasets[mode]
        dataset = parsing(nlp, dataset)

        save_file_name = os.path.join(args.output_dir, f'{mode}.json')
        print(f'save {len(dataset)} to', save_file_name)
        save_json(dataset, save_file_name)



def main2(args):
    nlp = spacy.load(args.spacy_model)

    dataset = list(load_json(args.data_dir))
    dataset = parsing(nlp, dataset)

    print(f'save {len(dataset)} to', args.output_dir)
    save_json(dataset, args.output_dir)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--spacy_model', default='en_core_web_sm', type=str)
    parser.add_argument('--data_dir',    type=str, default='data/origin')
    parser.add_argument('--dataset',     type=str, default='14res')
    parser.add_argument('--output_dir',  type=str, default='data/origin_syn')
    parser.add_argument('--batch_size',  default=32, type=int)
    parser.add_argument('--main2', action='store_true')

    args = parser.parse_args()
    args.data_dir   = os.path.join(args.data_dir,   args.dataset)
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    if args.main2:
        main2(args)
    else:
        main(args)


"""
python parsing.py --data_dir "./data/origin" --dataset 14res --output_dir "./data/orgin_syn"
python parsing.py --data_dir "./output/extraction/pseudo_labeled" --dataset yelp2023.json --output_dir "./output/extraction/pseudo_labeled_syn" --main2
"""
