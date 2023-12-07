import os
import time
import argparse
import torch
import pytorch_lightning as pl 
pl.seed_everything(42)


from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
from utils import load_json
from utils.triplet import make_triplets_seq, parse_triplets_seq
from utils.metric import F1_Measure



class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str='',
        max_seq_length: int = -1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        data_dir: str = '',
        dataset: str = '',
        seed: int = 42,
        test_as_dev: bool = False,
    ):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length     = max_seq_length
        self.train_batch_size   = train_batch_size
        self.eval_batch_size    = eval_batch_size
        self.data_dir           = os.path.join(data_dir, dataset)
        self.seed               = seed
        self.test_as_dev        = test_as_dev   

        print(self.data_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        
        if self.test_as_dev:
            dev_file_name   = os.path.join(self.data_dir, 'test.json')
        else:
            dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        self.raw_datasets = {
            'train': load_json(train_file_name),
            'dev'  : load_json(dev_file_name),
            'test' : load_json(test_file_name),
        }

        print('-----------data statistic-------------')
        print('Train', len(self.raw_datasets['train']))
        print('Dev',   len(self.raw_datasets['dev']))
        print('Test',  len(self.raw_datasets['test']))

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

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)



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

        batch_encodings = self.tok(text)
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        labels = self.make_labels(examples)

        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'examples'      : examples,
            'labels'        : labels,
        }

    def make_labels(self, examples):
        triplets_seqs = []
        for i in range(len(examples)):
            triplets_seq = self.make_triplets_seq(examples[i])
            triplets_seqs.append(triplets_seq)

        batch_encodings = self.tok(triplets_seqs)
        labels = batch_encodings['input_ids']
        labels = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100)
             for l in label]
            for label in labels
        ])

        return labels

    def make_triplets_seq(self, example):
        if 'triplets_seq' in example:
            return example['triplets_seq']

        triplets_seq = make_triplets_seq(example['triplets'])
        example['triplets_seq'] = triplets_seq
        return triplets_seq



class Result:
    def __init__(self, data):
        self.data = data 

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor >  other.monitor

    @classmethod
    def parse_from(cls, outputs):
        data = {}

        for output in outputs:
            examples = output['examples']
            predictions = output['predictions']

            for example, prediction in zip(examples, predictions):
                ID = example['ID']

                if 'triplets' in example:
                    triplets = []
                    for triplet in example['triplets']:
                        triplets.append((
                            triplet['aspect'][-1],
                            triplet['opinion'][-1],
                            triplet['sentiment'],
                        ))
                else:
                    raise NotImplementedError

                triplet_preds = parse_triplets_seq(prediction, example['sentence'])

                data[ID] = {
                    'ID': ID,
                    'sentence': example['sentence'],
                    'triplets': triplets,
                    'triplet_preds': triplet_preds,
                }

        return cls(data)

    def cal_metric(self):
        f1 = F1_Measure()

        for ID in self.data:
            example = self.data[ID]
            g = example['triplets']
            p = example['triplet_preds']
            f1.true_inc(ID, g)
            f1.pred_inc(ID, p)

        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['r'],
            'precision': f1['p'],
        }

        self.monitor = self.detailed_metrics['f1']

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()




class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
     
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path
        )

    def _make_model_dir(self):
        return os.path.join(self.hparams.output_dir, 'model', f'dataset={self.hparams.dataset},b={self.hparams.output_sub_dir},seed={self.hparams.seed}')

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
        examples = inputs.pop('examples')
        outputs  = self.model(**inputs)
    
        return {
            'examples': examples,
            'loss': outputs[0]
        }

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            num_return_sequences=1,
            max_length=100,
            num_beams=1,
        )

        generateds = self.data_module.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return {
            'examples': batch['examples'],
            'predictions': generateds
        }

    def validation_epoch_end(self, outputs):
        # examples = self.data_module.raw_datasets['dev']
        self.current_val_result = Result.parse_from(outputs)
        self.current_val_result.cal_metric()

        if not hasattr(self, 'best_val_result'):
            self.best_val_result = self.current_val_result
            self.save_model()

        elif self.best_val_result < self.current_val_result:
            self.best_val_result = self.current_val_result
            self.save_model()  

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.test_result = Result.parse_from(outputs)
        self.test_result.cal_metric()

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
    def on_validation_end(self, trainer, pl_module):
        print()
        if hasattr(pl_module, 'current_train_result'):
            pl_module.current_train_result.report()
        print()
        print('------------------------------------------------------------')
        print('[current]', end=' ')
        pl_module.current_val_result.report()

        print('[best]   ', end=' ')
        pl_module.best_val_result.report()
        print('------------------------------------------------------------\n')

    def on_test_end(self, trainer, pl_module):
        pl_module.test_result.report()




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
        'num_sanity_val_steps': 5 if args.do_train else 0,
        'enable_checkpointing': False,
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