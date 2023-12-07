import os
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl 
pl.seed_everything(42)

from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils import load_json, save_json
from utils.triplet import make_triplets_seq
from utils.metric import FluencyScorer



class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str='',
        max_seq_length: int = -1,
        eval_batch_size: int = 32,
        data_dir: str = '',
        train_data_dir: str = '',
        seed: int = 42,
    ):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length     = max_seq_length
        self.eval_batch_size    = eval_batch_size
        self.data_dir           = data_dir
        self.train_data_dir     = train_data_dir
        self.seed               = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def load_dataset(self):
        dev_file_name   = os.path.join(self.data_dir, 'train.json')
        test_file_name  = os.path.join(self.data_dir, 'train.json')

        train_file_name = self.train_data_dir
        train_examples = [
            example for example in load_json(train_file_name)
            if example['min_con'] > 0.75 and example['avg_con'] > 0.95
        ]

        print('--------- data statistic ---------')
        for thre in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
            _examples = [
                example for example in load_json(train_file_name)
                if example['min_con'] > thre
            ]
            print(thre, len(_examples))

        self.raw_datasets = {
            'train': self.make_example(train_examples),
            'dev'  : self.make_example(load_json(dev_file_name), repk=4),
            'test' : self.make_example(load_json(test_file_name), repk=4),
        }

        print('--------- data statistic ---------')
        print('train:', len(self.raw_datasets['train']))
        print('dev:',   len(self.raw_datasets['dev']))
        print('test:',  len(self.raw_datasets['test']))
        print()

    def make_example(self, examples, repk=1):
        new_examples = []
        for example in examples:
            for i in range(repk):
                new_example = {
                    'ID': example['ID'],
                    'sentence': example['sentence'],
                    'syn_seq' : example['syn_seq'],
                }
                random_example = random.choice(examples)

                if 'triplets' in random_example:
                    new_example['triplets'] = random_example['triplets']
                    new_example['origin_triplets'] = example['triplets']

                elif 'triplets_seq' in random_example:
                    new_example['triplets_seq'] = random_example['triplets_seq']
                    new_example['origin_triplets_seq'] = example['triplets_seq']

                new_examples.append(new_example)

        return new_examples

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=8,
            num_workers=1,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer, 
                max_seq_length=self.max_seq_length
            )
        )

        return dataloader

    def predict_dataloader(self):

        train_dataloader = self.get_dataloader('train', self.eval_batch_size, shuffle=False)
        dev_dataloader   = self.get_dataloader('dev', self.eval_batch_size, shuffle=False)
        test_dataloader  = self.get_dataloader('test', self.eval_batch_size, shuffle=False)

        return [train_dataloader, dev_dataloader, test_dataloader]



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
            triplets_seq = self.make_triplets_seq(example)
            # make origin_triplets_seq, and save to example
            self.make_triplets_seq(example, key1='origin_triplets_seq', key2= 'origin_triplets')
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

    def make_triplets_seq(self, example, key1='triplets_seq', key2='triplets'):
        if key1 in example:
            return example[key1]

        triplets_seq  = make_triplets_seq(example[key2])
        example[key1] = triplets_seq
        return triplets_seq




class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        print(self.hparams.model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

    def decode(self, ids):
        return self.data_module.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_beams=1,
            num_return_sequences=1,
            do_sample=True,
            top_k=0,
            top_p=1,
        )

        generateds = self.decode(generated_ids)
        examples = batch['examples']
        
        return {
            'generateds': generateds,
            'examples': examples,
            'dataloader_idx': dataloader_idx,
        }

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--output_dir", type=str)

        return parser



class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval='epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.flu_scorer = FluencyScorer('/data/zhangyice/2023/pretrained_models/parrot_fluency_model/')

    def compute_flu(self, text):
        flu = self.flu_scorer.compute_flu(text, bz=100, progress_bar=False)
        return flu

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for predictions_i, mode in zip(predictions, ['train', 'dev', 'test']):
            output_examples = []

            for output in tqdm(predictions_i, desc=mode):
                generateds = output['generateds']
                examples   = output['examples']

                origin_sentences = [example['sentence'] for example in examples]
                examples = output['examples']

                flus_1 = self.compute_flu(origin_sentences)
                flus_2 = self.compute_flu(generateds)

                for generated, example, flu1, flu2 in zip(generateds, examples, flus_1, flus_2):

                    output_example = {
                        'ID': len(output_examples),
                        'triplets_seq': example['triplets_seq'],
                        'syn_seq'     : example['syn_seq'],
                        'origin_triplets_seq': example['origin_triplets_seq'],
                        'generated_sentence' : generated,
                        'origin_sentence'    : example['sentence'],
                        'genera_flu': flu2,
                        'origin_flu': flu1,
                    }
                    output_examples.append(output_example)

            output_dir = os.path.join(self.output_dir, f'{mode}.json')
            print(f'save {len(output_examples)} to', output_dir)
            save_json(output_examples, output_dir)



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

    predictions = trainer.predict(
        model, 
        datamodule=data_module, 
        return_predictions=False
    )


if __name__ == '__main__':
    main()