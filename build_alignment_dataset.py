import os
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl 
pl.seed_everything(42)

import re 
import spacy 
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
from utils import load_json, save_json
from utils.triplet import make_triplets_seq



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
            # if example['min_con'] > 0.75 and example['avg_con'] > 0.95
            # for response
            if example['min_con'] > 0.5 and example['avg_con'] > 0.95
        ]

        print('--------- data statistic ---------')
        for thre in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.60, 0.55, 0.5):
            _examples = [
                example for example in load_json(train_file_name)
                if example['min_con'] > thre
            ]
            print(thre, len(_examples))

        self.raw_datasets = {
            'train': train_examples,
            'dev'  : load_json(dev_file_name),
            'test' : load_json(test_file_name),
        }

        print('--------- data statistic ---------')
        print('train:', len(self.raw_datasets['train']))
        print('dev:',   len(self.raw_datasets['dev']))
        print('test:',  len(self.raw_datasets['test']))
        print()

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
        text = [example['sentence'] for example in examples]
        for example in examples:
            self.make_triplets_seq(example)

        batch_encodings = self.tok(text)
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'examples'      : examples
        }

    def make_triplets_seq(self, example):
        if 'triplets_seq' in example:
            return example['triplets_seq']

        triplets_seq = make_triplets_seq(example['triplets'])
        example['triplets_seq'] = triplets_seq
        return triplets_seq



class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
     
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path
        )

    def decode(self, ids):
        return self.data_module.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_beams=4,
            num_return_sequences=4,
        )

        generateds = self.decode(generated_ids)
        generateds_beam = []
        for i in range(len(generateds)//4):
            generateds_beam.append(generateds[i*4: i*4+4])

        return {
            'triplets_seqs_beam': generateds_beam,
            'examples'          : batch['examples'],
        }

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--output_dir", type=str)

        return parser




class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval='epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for predictions_i, mode in zip(predictions, ['train', 'dev', 'test']):
            output_examples = []

            for output in tqdm(predictions_i, desc=mode):

                triplets_seqs_beam = output['triplets_seqs_beam']
                examples = output['examples']

                for triplets_seq_beam, example in zip(triplets_seqs_beam, examples):

                    output_example = {
                        'ID': example['ID'],
                        'sentence': example['sentence'],
                        'triplets_seq': example['triplets_seq'],
                        'triplets_seq_beam': triplets_seq_beam,
                        'min_con': example['min_con'] if 'min_con' in example else None,
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