import os
import random
import math
import argparse
from tqdm import tqdm
import torch
import pytorch_lightning as pl 
pl.seed_everything(42)


from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import DataLoader
from collections import defaultdict

from model.t5_for_classification import T5ForClassification
from utils import save_json, load_json
from utils.triplet import make_triplets_seq



def get_synseq_len(syn_seq):
    return syn_seq.count(' ')+1


def get_triplet_num_aspect_num_opinion_num(triplets_or_triplets_seq):
    aspects  = []
    opinions = []

    if type(triplets_or_triplets_seq) is str:
        triplet_num = triplets_or_triplets_seq.count(';') + 1
        for triplet_seq in triplets_or_triplets_seq.split(';'):
            aspect = triplet_seq.split('|')[0].strip()
            aspects.append(aspect)
            
            opinion = triplet_seq.split('|')[1].strip()
            opinions.append(opinion)
    else:
        triplet_num = len(triplets_or_triplets_seq)
        for triplet in triplets_or_triplets_seq:
            aspect = triplet['aspect'][-1]
            aspects.append(aspect)
            
            opinion = triplet['opinion'][-1]
            opinions.append(opinion)

    return triplet_num, len(aspects), len(opinions)



class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 generator_path: str='',
                 max_seq_length: int = -1,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 reference_data_dir: str = '',
                 seed: int = 42,
                 num_augment_example: int = 12660,
                 sample_triplet_from_where: str = 'reference'):

        super().__init__()
        self.generator_path     = generator_path
        self.max_seq_length     = max_seq_length
        self.eval_batch_size    = eval_batch_size
        self.data_dir           = data_dir
        self.reference_data_dir = reference_data_dir
        self.seed               = seed
        self.num_augment_example= num_augment_example
        self.sample_triplet_from_where = sample_triplet_from_where

        self.tokenizer = AutoTokenizer.from_pretrained(generator_path, use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        self.train_examples = load_json(train_file_name)

        self.reference_examples = [
            example for example in load_json(self.reference_data_dir)
            if example['min_con'] > 0.75 and example['avg_con'] > 0.95
        ]

        self.index_example_by_tao(self.reference_examples)
        self.augmented_examples = self.make_example(num_example=self.num_augment_example)


        print('-----------data statistic-------------')
        print('Train',     len(self.train_examples))
        print('Reference', len(self.reference_examples))
        print('Augmented', len(self.augmented_examples))

        print('Train')
        self.stat_triplet_num(self.train_examples)
        print('Reference')
        self.stat_triplet_num(self.reference_examples)
        print('Augmented')
        self.stat_triplet_num(self.augmented_examples)

    def stat_triplet_num(self, examples):
        counter = defaultdict(int)
        for example in examples:
            triplets_seq = (example['triplets_seq'] if 'triplets_seq' in example
                            else make_triplets_seq(example['triplets']))
            tao = get_triplet_num_aspect_num_opinion_num(triplets_seq)
            triplet_num = tao[0]
            counter[triplet_num] += 1

        for triplet_num in sorted(counter.keys()):
            print(triplet_num, ':', counter[triplet_num])

    def index_example_by_tao(self, examples):
        self.example_indexed_by_tao = defaultdict(list)

        for example in examples:
            triplets_seq = (example['triplets_seq'] if 'triplets_seq' in example
                            else make_triplets_seq(example['triplets']))
            tao = get_triplet_num_aspect_num_opinion_num(triplets_seq)

            self.example_indexed_by_tao[tao].append({
                'syn_seq': example['syn_seq'], 
                'triplets_seq': triplets_seq, 
                'sentence': example['sentence']
            })

        print('-----------syn statistic-------------')
        for triplet_num in range(1, 7+1):
            examples = []
            for tao in self.example_indexed_by_tao:
                if tao[0] == triplet_num:
                    _examples = self.example_indexed_by_tao[tao]
                    examples.extend(_examples)

            print(triplet_num, len(examples), 
                  sum([get_synseq_len(example['syn_seq']) for example in examples]) / len(examples))

    def make_example(self, num_example):
        new_examples = []
        num_train_examples = len(self.train_examples)

        for i in range(num_example):
            example = self.train_examples[i%num_train_examples]
            triplets_seq = make_triplets_seq(example['triplets'])
            tao = get_triplet_num_aspect_num_opinion_num(triplets_seq)

            if self.sample_triplet_from_where == 'train':
                m_triplets_seq = make_triplets_seq
                triplet_sentence = example['sentence']

            elif self.sample_triplet_from_where == 'reference':
                m_triplets_seq, triplet_sentence = self.sample_triplets_by_tao(tao)
                tao = get_triplet_num_aspect_num_opinion_num(m_triplets_seq)

            syn_seq, syn_sentence = self.sample_syn_by_tao(tao)
            new_example = {
                'ID': f"{i}",
                'triplets_seq': m_triplets_seq,
                'triplet_sentence': triplet_sentence,
                'syn_seq': syn_seq,
                'syn_sentence': syn_sentence,
            }

            new_examples.append(new_example)

        return new_examples

    def sample_syn_by_tao(self, tao):
        examples = self.get_examples_by_tao(tao)
        random_example = random.choice(examples)
        return random_example['syn_seq'], random_example['sentence']

    def sample_triplets_by_tao(self, tao):
        examples = self.get_examples_by_tao(tao)
        random_example = random.choice(examples)
        return random_example['triplets_seq'], random_example['sentence']
    
    def get_examples_by_tao(self, tao):
        if tao in self.example_indexed_by_tao:
            examples = self.example_indexed_by_tao[tao]
        else:
            most_similarity_tao = min([(abs(tao[0]-t), abs(tao[1]-a), abs(tao[2]-o), (t,a,o)) for t,a,o in self.example_indexed_by_tao.keys()])[-1]
            examples = self.example_indexed_by_tao[most_similarity_tao]
        return examples

    def predict_dataloader(self):
        dataloader = DataLoader(
            dataset=self.augmented_examples,
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=8,
            num_workers=16,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer, 
                max_seq_length=self.max_seq_length,
            )
        )
        return dataloader



def tok(tokenizer, text, max_seq_length):
    kwargs = {
        'text': text,
        'return_tensors': 'pt'
    }

    if max_seq_length in (-1, 'longest'):
        kwargs['padding'] = True

    else:
        kwargs['padding'] = True
        kwargs['truncation'] = True
        kwargs['max_length'] = max_seq_length

    batch_encodings = tokenizer(**kwargs)
    return batch_encodings




class DataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def tok(self, text):
        return tok(self.tokenizer, text, self.max_seq_length)

    def __call__(self, examples):
        conditions = []

        for example in examples:
            syn_seq = example['syn_seq'].lower()
            triplets_seq = example['triplets_seq']
            condition = triplets_seq + ' ; ' + syn_seq
            conditions.append(condition)

        batch_encodings = self.tok(conditions)
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'examples': examples
        }



class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        
        if not self.hparams.no_generate:
            self.generator = T5ForConditionalGeneration.from_pretrained(self.hparams.generator_path)
        self.extractor = T5ForConditionalGeneration.from_pretrained(self.hparams.extractor_path)

        self.alignment_model = T5ForClassification.from_pretrained(self.hparams.alignment_model_path)
        self.fluency_model   = T5ForClassification.from_pretrained(self.hparams.fluency_model_path)

    def generate(self, batch):

        generated_ids = self.generator.generate(
             input_ids=batch['input_ids'], 
             attention_mask=batch['attention_mask'],
             num_return_sequences=1,
             max_length=self.hparams.max_seq_length,
             num_beams=1,
             do_sample=True,
             top_k=0,
             top_p=1,
        )
        generateds = self.decode(generated_ids)
        
        batch_encodings = self.tok(generateds)
        triplets_preds  = self.extractor.generate(
            input_ids=batch_encodings['input_ids'].cuda(),
            attention_mask=batch_encodings['attention_mask'].cuda(),
            num_return_sequences=1,
            max_length=self.hparams.max_seq_length,
            num_beams=1,
        )

        triplets_preds = self.decode(triplets_preds)
        alignment_scores, fluency_scores = self.cal_scores(generateds, batch)

        return {
            'generateds': generateds,
            'triplets_preds': triplets_preds,
            'alignment_scores': alignment_scores,
            'fluency_scores': fluency_scores,
            'examples': batch['examples'],
        }

    def cal_scores(self, generateds, batch):
        triplets_seqs = [example['triplets_seq'] for example in batch['examples']]
        triplets_generateds = [triplets_seq + ' ; ' + generated 
                               for triplets_seq, generated in zip(triplets_seqs, generateds)]

        batch_encodings = self.tok(triplets_generateds, max_length=140)
        alignment_scores = self.alignment_model(
            input_ids=batch_encodings['input_ids'].cuda(),
            attention_mask=batch_encodings['attention_mask'].cuda(),
        )
        
        batch_encodings = self.tok(generateds)
        fluency_scores = self.fluency_model(
            input_ids=batch_encodings['input_ids'].cuda(),
            attention_mask=batch_encodings['attention_mask'].cuda(),
        )
        return alignment_scores, fluency_scores

    def no_generate(self, batch):

        sentences = [example['triplet_sentence'] for example in batch['examples']]
        alignment_scores, fluency_scores = self.cal_scores(sentences, batch)

        return {
            'generateds': sentences,
            'triplets_preds': [''] * len(sentences),
            'alignment_scores': alignment_scores,
            'fluency_scores': fluency_scores,
            'examples': batch['examples'],
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if hasattr(self, 'generator'):
            self.generator.eval()

        self.extractor.eval()
        self.alignment_model.eval()
        self.fluency_model.eval()

        with torch.no_grad():
            if self.hparams.no_generate:
                output = self.no_generate(batch)
            else:    
                output = self.generate(batch)

        output['dataloader_idx'] = dataloader_idx

        return output

    def decode(self, input_ids):
        return self.data_module.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def tok(self, text, max_length=None):
        max_length = self.hparams.max_seq_length if max_length is None else max_length
        return tok(self.data_module.tokenizer, text, max_length)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--extractor_path", type=str)
        parser.add_argument("--alignment_model_path", type=str)
        parser.add_argument("--fluency_model_path", type=str)
        parser.add_argument("--no_generate", action='store_true')
        
        return parser



class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval='epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        output_examples = []

        for output in tqdm(predictions[0]):

            examples = output['examples']
            generateds = output['generateds']
            triplets_preds = output['triplets_preds']
            alignment_scores = output['alignment_scores']
            fluency_scores = output['fluency_scores']

            for example, generated, triplets_pred, alignment_score, fluency_score in zip(examples, generateds, triplets_preds, alignment_scores, fluency_scores):

                example = {
                    'ID': len(output_examples),
                    'triplets_seqq': example['triplets_seq'],
                    'triplets_pred': triplets_pred,
                    'syn_seq': example['syn_seq'],
                    'generated_sentence': generated,
                    'triplet_sentence': example['triplet_sentence'],
                    'syn_sentence': example['syn_sentence'],
                    'alignment_score': float(alignment_score),
                    'fluency_score': float(fluency_score),
                }

                output_examples.append(example)

        output_file_name = os.path.join(self.output_dir, 'augmented_train.json')
        print(f'save {len(output_examples)} to', output_file_name)
        save_json(output_examples, output_file_name)



def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LightningModule.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    data_module = DataModule.from_argparse_args(args)
    data_module.load_dataset()

    model = LightningModule(args, data_module)

    pred_writer = CustomWriter(output_dir=args.output_dir)
    kwargs = {
        'callbacks': [pred_writer],
        'logger': False,
        'enable_checkpointing': False,
    }
    trainer = pl.Trainer.from_argparse_args(args, **kwargs)

    trainer.predict(model, datamodule=data_module, return_predictions=False)



if __name__ == '__main__':
    main()
