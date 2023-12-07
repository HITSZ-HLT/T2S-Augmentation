
import os
import time
import random
import math
import argparse
import torch
import difflib
import pytorch_lightning as pl 
pl.seed_everything(42)

import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model.t5_for_classification import T5ForClassification
from utils import load_json, tgenerate_batch, generate_batch
from utils.triplet import make_triplets_seq



class DataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path: str='',
                 max_seq_length: int = -1,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 seed: int = 42):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length     = max_seq_length
        self.train_batch_size   = train_batch_size
        self.eval_batch_size    = eval_batch_size
        self.data_dir           = data_dir
        self.seed               = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        self.raw_datasets = {
            'train': load_json(train_file_name),
            'dev'  : load_json(dev_file_name),
            'test' : load_json(test_file_name)
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
                max_seq_length=self.max_seq_length,
                mode=mode,
            )
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader('dev', self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader('test', self.eval_batch_size, shuffle=False)



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

    def __init__(self, tokenizer, max_seq_length, mode):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

    def tok(self, text):
        return tok(self.tokenizer, text, self.max_seq_length)

    def __call__(self, examples):
        conditions = []
        real_samples = []
        fake_samples = []
        fake_type_ids  = []

        for example in examples:
            triplets_seq = example['triplets_seq']
            real_sample  = triplets_seq + ' ; ' + example['sentence']
            real_samples.append(real_sample)

            negatives = [
                (i, negative) 
                for i, negative in enumerate(example['triplets_seq_beam']) 
                if negative != triplets_seq
            ]

            if self.mode == 'train':

                if len(negatives) > 0:
                    i, negative = random.choice(negatives)
                    fake_sample = negative + ' ; ' + example['sentence']
                    fake_samples.append(fake_sample)
                    fake_type_ids.append(i)

            else:
                for i, negative in negatives:
                    fake_sample = negative + ' ; ' + example['sentence']
                    fake_samples.append(fake_sample)
                    fake_type_ids.append(i)

        real_batch_encodings = self.tok(real_samples)
        fake_batch_encodings = self.tok(fake_samples)

        real_input_ids = real_batch_encodings['input_ids']
        fake_input_ids = fake_batch_encodings['input_ids']
        real_attention_mask = real_batch_encodings['attention_mask']
        fake_attention_mask = fake_batch_encodings['attention_mask']

        fake_type_ids  = torch.tensor(fake_type_ids)

        return {
            'real_input_ids': real_input_ids,
            'fake_input_ids': fake_input_ids,
            'real_attention_mask': real_attention_mask,
            'fake_attention_mask': fake_attention_mask,
            'fake_type_ids': fake_type_ids
        }



class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        self.model = T5ForClassification.from_pretrained(self.hparams.model_name_or_path)
    
    def _make_model_dir(self):
        return os.path.join(self.hparams.output_dir, 'model', f'b={self.hparams.output_sub_dir}')

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = self._make_model_dir()
        print(f'## save model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)
        self.data_module.tokenizer.save_pretrained(dir_name)

    def load_model(self):
        dir_name = self._make_model_dir()
        print(f'## load model to {dir_name}')
        self.model = T5ForClassification.from_pretrained(dir_name)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), 
            eps=self.hparams.adam_epsilon, 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    def decode(self, ids):
        return self.data_module.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def forward(self, **batch):
        fake_type_ids = batch.pop('fake_type_ids')
        output = self.model.forward_train(**batch)
        output['fake_type_ids'] = fake_type_ids
        return output
    
    def training_step(self, batch, batch_idx):
        output = self(**batch)

        loss  = output['loss']
        accu1 = output['accu1']
        accu2 = output['accu2']
        accu  = output['accu']

        self.log('accu1', accu1, prog_bar=True)
        self.log('accu2', accu2, prog_bar=True)
        self.log('accu',  accu,  prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)

        fake_type_ids = output['fake_type_ids']
        real_score = output['real_score']
        fake_score = output['fake_score']

        fak1_acc = (fake_score[fake_type_ids==1]<0).float().mean()
        fak2_acc = (fake_score[fake_type_ids==2]<0).float().mean()
        fak3_acc = (fake_score[fake_type_ids==3]<0).float().mean()
        
        return {
            'loss' : output['loss'].item(), 
            'accu' : output['accu'].item(),
            'fake_sentences': self.decode(output['fake_sentences']),
            'real_sentences': self.decode(output['real_sentences']),
            'fake_score': fake_score,
            'real_score': real_score,
            'real_acc': output['accu1'],
            'fake_acc': output['accu2'],
            'fak1_acc': fak1_acc,
            'fak2_acc': fak2_acc,
            'fak3_acc': fak3_acc,
        }

    def eval_epoch_end(self, outputs):
        loss = sum([output['loss'] for output in outputs]) / len(outputs)
        accu = sum([output['accu'] for output in outputs]) / len(outputs)

        real_acc = sum([output['real_acc'] for output in outputs]) / len(outputs)
        fake_acc = sum([output['fake_acc'] for output in outputs]) / len(outputs)
        fak1_acc = sum([output['fak1_acc'] for output in outputs]) / len(outputs)
        fak2_acc = sum([output['fak2_acc'] for output in outputs]) / len(outputs)
        fak3_acc = sum([output['fak3_acc'] for output in outputs]) / len(outputs)
        
        metric = {
            'loss' : loss,
            'accu' : accu,
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'fak1_acc': fak1_acc,
            'fak2_acc': fak2_acc,
            'fak3_acc': fak3_acc,
            'monitor': accu,
        }
        return metric

    def validation_epoch_end(self, outputs):
        self.current_val_metric = self.eval_epoch_end(outputs)

        if not hasattr(self, 'best_val_metric'):
            self.best_val_metric = self.current_val_metric
            self.save_model()

        elif self.best_val_metric['monitor'] < self.current_val_metric['monitor']:
            self.best_val_metric = self.current_val_metric
            self.save_model()

    def test_step(self, batch, batch_idx):

        output = self.validation_step(batch, batch_idx)
        for fsent, fscore, rsent, rscore in zip(
            output['fake_sentences'],
            output['fake_score'],
            output['real_sentences'],
            output['real_score']
        ):  

            if rscore < 0:
                print('real:', rsent, rscore.item())
                print()

            if fscore > 0:
                print('fake:', fsent, fscore.item())
                print()

        return output

    def test_epoch_end(self, outputs):
        self.test_metric = self.eval_epoch_end(outputs)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0., type=float)

        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--output_sub_dir", type=str)
        parser.add_argument("--do_train", action='store_true')

        return parser



class LoggingCallback(pl.Callback):
    
    def print_dict(self, prefix, dic):
            print(prefix + ' | '.join([f'{k}: {v:.4f}' for k,v in dic.items()]))

    def on_validation_end(self, trainer, pl_module):
        print()
        self.print_dict('[current] ', pl_module.current_val_metric)
        self.print_dict('[best]    ', pl_module.best_val_metric)
        print()

    def on_test_end(self, trainer, pl_module):
        print()
        self.print_dict('[test]', pl_module.test_metric)
        print()



def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LightningModule.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    data_module = DataModule.from_argparse_args(args)
    data_module.load_dataset()

    model = LightningModule(args, data_module)

    logging_callback = LoggingCallback()  
    kwargs = {
        'callbacks': [logging_callback],
        'logger': False,
        'enable_checkpointing': False,
        'num_sanity_val_steps': 5 if args.do_train else 0,
    }
    trainer = pl.Trainer.from_argparse_args(args, **kwargs)

    if args.do_train:
        trainer.fit(model, datamodule=data_module)
        model.load_model()
        trainer.test(model, datamodule=data_module)

    else:
        model.load_model()
        trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()