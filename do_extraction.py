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
from utils import load_line_json, save_json, tgenerate_batch
from utils.triplet import make_triplets_seq, parse_triplets_seq



def text_len(text):
    return text.count(' ')+1


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str='',
        max_seq_length: int = -1,
        eval_batch_size: int = 32,
        data_dir: str = '',
        seed: int = 42,
    ):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length     = max_seq_length
        self.eval_batch_size    = eval_batch_size
        self.data_dir           = data_dir
        self.seed               = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def load_dataset(self):

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('sentencizer')

        min_length = 5
        max_length = 100

        dataset = list(load_line_json(self.data_dir))

        predict_examples = []
        for batch_examples in tgenerate_batch(dataset, bz=32):

            texts = [example['Text'] for example in batch_examples]
            docs  = nlp.pipe(texts, disable=['tagger', 'tok2vec', 'parser', 'lemmatizer', 'ner'])

            for doc, example in zip(docs, batch_examples):
                for i, sentence in enumerate(doc.sents):
                    sentence = str(sentence).strip()
                    sentence = sentence.replace('\r', '')
                    # '(good)' -> '( good )'
                    sentence = re.sub(r'\((?P<v1>[^ ])(?P<v2>.*)(?P<v3>[^ ])\)', lambda x: '( ' + x.group('v1') + x.group('v2') + x.group('v3') + ' )', sentence)

                    if not (min_length <= text_len(sentence) <= max_length):
                        continue

                    new_example = {
                        'ID': f"{example['ID']}-{i+1}",
                        'sentence': sentence
                    }
                    predict_examples.append(new_example)

        self.raw_dataset = predict_examples
        print('-----------data statistic-------------')
        print('Predict', len(self.raw_dataset))

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.raw_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=8,
            num_workers=1,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer, 
                max_seq_length=self.max_seq_length
            )
        )



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

        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'examples'      : examples,
        }



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
        generated_outputs = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            num_return_sequences=1,
            max_length=100,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True
        )

        generateds = self.decode(generated_outputs['sequences'])
        confidences = self.get_confidence(generated_outputs)

        return {
            'examples': batch['examples'],
            'predictions': generateds,
            'confidences': confidences
        }

    def get_mask(self, input_ids):
        eos_token_id = self.model.config.eos_token_id
        pad_token_id = self.model.config.pad_token_id

        eos_flag = (input_ids == eos_token_id)
        eos_flag = torch.cat([eos_flag[:, :1], eos_flag[:, :-1]], dim=1)
        attention_mask = torch.cumsum(eos_flag, dim=1)
        attention_mask = (attention_mask == 0).bool()

        return attention_mask.long()

    def get_confidence(self, generated_outputs):
        input_ids = generated_outputs['sequences']
        attention_mask = self.get_mask(input_ids)[:, 1:] # 1: to remove decoder_start_id

        probs = torch.stack(generated_outputs.scores, dim=1)
        probs = F.log_softmax(probs, dim=-1)
        confidence = probs.max(dim=-1)[0]

        confidence[~attention_mask.bool()] = 0
        min_confidence = confidence.min(dim=-1)[0].exp().detach().cpu().numpy()

        avg_confidence = confidence.sum(dim=-1) / attention_mask.sum(dim=-1)
        avg_confidence = avg_confidence.exp().detach().cpu().numpy()

        return min_confidence, avg_confidence

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--output_dir", type=str)

        return parser



class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval='epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):

        output_examples = []
        N = 0
        for output in tqdm(predictions[0]):
            examples = output['examples']
            predictions = output['predictions']
            min_confidence, avg_confidence = output['confidences']

            for example, prediction, min_con, avg_con in zip(examples, predictions, min_confidence, avg_confidence):

                N += 1
                if min_con >= 0.7 and avg_con >= 0.9 and self.check_triple(example, prediction):
                    new_example = {
                        'ID': example['ID'],
                        'sentence': example['sentence'],
                        'triplets_seq': prediction,
                        'min_con' : float(min_con),
                        'avg_con' : float(avg_con),
                    }
                    output_examples.append(new_example)

        print(f'save {len(output_examples)}/{N} to', self.output_dir)
        save_json(output_examples, self.output_dir)

    def check_triple(self, example, prediction):
        return parse_triplets_seq(
            prediction, 
            example['sentence'], 
            if_not_complete_drop_all=True
        )


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

    pred_writer = CustomWriter(output_dir=os.path.join(args.output_dir))
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