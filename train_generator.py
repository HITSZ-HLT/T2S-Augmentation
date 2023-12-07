import os
import random
import time
import argparse
import torch
import pytorch_lightning as pl
pl.seed_everything(42)


from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration

from utils import load_json
from utils.metric import compute_bleu, compute_rouge, compute_meteor
from utils.metric import FluencyScorer
from utils.triplet import make_triplets_seq



class DataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path: str='',
                 max_seq_length1: int = -1,
                 max_seq_length2: int = -1,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 train_data_dir: str = '',
                 seed: int = 42):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length1    = max_seq_length1
        self.max_seq_length2    = max_seq_length2
        self.train_batch_size   = train_batch_size
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
            'train': train_examples,
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
                max_seq_length1=self.max_seq_length1,
                max_seq_length2=self.max_seq_length2,
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
    def __init__(self, tokenizer, max_seq_length1, max_seq_length2, mode):
        self.tokenizer = tokenizer
        self.max_seq_length1 = max_seq_length1
        self.max_seq_length2 = max_seq_length2
        self.mode = mode

    def tok(self, text, max_seq_length):
        return tok(self.tokenizer, text, max_seq_length)

    def make_labels(self, examples):
        text = [example['sentence'] for example in examples]
        batch_encodings = self.tok(text, self.max_seq_length2)

        labels = batch_encodings['input_ids']
        labels = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100)
             for l in label]
            for label in labels
        ])

        return labels

    def make_inputs(self, examples):
        conditions = []
        for example in examples:
            syn_seq = example['syn_seq']
            syn_seq = syn_seq.lower()

            triplets_seq = self.make_triplets_seq(example)
            condition = triplets_seq + ' ; ' + syn_seq
            conditions.append(condition)

        batch_encodings = self.tok(conditions, self.max_seq_length1)
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        return input_ids, attention_mask

    def make_triplets_seq(self, example):
        if 'triplets_seq' in example:
            return example['triplets_seq']

        triplets_seq = make_triplets_seq(example['triplets'])
        example['triplets_seq'] = triplets_seq
        return triplets_seq

    def __call__(self, examples):
        input_ids, attention_mask = self.make_inputs(examples)
        labels = self.make_labels(examples)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }



class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.fluency_scorer = FluencyScorer()

        # print(self.model.config)

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
        self.model = T5ForConditionalGeneration.from_pretrained(dir_name)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]

        self.log('train_loss', loss)
        return loss

    def decode(self, ids):
        return self.data_module.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def generate(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        generated_ids = self.model.generate(
            input_ids, 
            attention_mask=attention_mask,
            num_return_sequences=1, 
            max_length=self.hparams.max_seq_length2
        )
        conditions = self.decode(input_ids)
        generateds = self.decode(generated_ids)
        
        labels = batch['labels']
        labels[labels==-100] = self.data_module.tokenizer.pad_token_id
        labels = self.decode(labels)
        
        return conditions, generateds, labels

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]

        conditions, generateds, labels = self.generate(batch)

        bleus  = compute_bleu(labels, generateds)
        rouges = compute_rouge(labels, generateds)
        meteor = compute_meteor(labels, generateds)
        flus   = self.fluency_scorer.compute_flu(generateds, progress_bar=False)

        self.log('valid_loss', loss)

        return {
            'loss'  : loss.item(),
            'bleus' : bleus,
            'rouges': rouges,
            'meteor': meteor,
            'flus'  : flus
        }

    def eval_epoch_end(self, outputs):
        loss  = sum([output['loss'] for output in outputs]) / len(outputs)
        
        bleu1, bleu2, bleu3, bleu4, rouge_l, meteor, flus = [], [], [], [], [], [], []
        for output in outputs:
            for bleus in output['bleus']:
                bleu1.append(bleus[0])
                bleu2.append(bleus[1])
                bleu3.append(bleus[2])
                bleu4.append(bleus[3])
            
            for rouges in output['rouges']:
                rouge_l.append(rouges['rouge-l']['r'])

            meteor.extend(output['meteor'])
            flus.extend(output['flus'])

        bleu1   = sum(bleu1) / len(bleu1)
        bleu2   = sum(bleu2) / len(bleu2)
        bleu3   = sum(bleu3) / len(bleu3)
        bleu4   = sum(bleu4) / len(bleu4)
        rouge_l = sum(rouge_l) / len(rouge_l)
        meteor  = sum(meteor) / len(meteor)
        flu     = sum(flus) / len(flus)

        return {
            'loss' : loss,
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu3': bleu3,
            'bleu4': bleu4,
            'rouge_l': rouge_l * 100,
            'meteor' : meteor * 100,
            'flu'    : flu,
            'monitor': (bleu4+rouge_l*100+meteor*100)/3
        }

    def validation_epoch_end(self, outputs):
        self.current_val_metric = self.eval_epoch_end(outputs)

        if not hasattr(self, 'best_val_metric'):
            self.best_val_metric = self.current_val_metric
            self.save_model()

        elif self.best_val_metric['monitor'] < self.current_val_metric['monitor']:
            self.best_val_metric = self.current_val_metric
            self.save_model()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

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
        self.print_dict('[test] ', pl_module.test_metric)
        


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
        'num_sanity_val_steps': 2 if args.do_train else 0,
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