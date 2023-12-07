import os
import time
import random
import math
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl 
pl.seed_everything(42)

from collections import defaultdict
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence

from utils import load_json
from utils.triplet import make_triplets_seq
from utils.trlx_utils import RunningMoments, whiten, logprobs_of_labels, masked_mean

from model.t5_for_classification import T5ForClassification
# from model.t5_for_classification import T5ForClassification as T5ForClassification2
# from model.t5_for_classification2 import T5ForClassification as T5ForClassification2
from transformers import T5ForConditionalGeneration




def make_head(n_embd: int, out: int, dtype: type = torch.float32) -> nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2, dtype=dtype),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out, dtype=dtype),
    )


class ActorModelWithValueHead(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_head = make_head(self.config.d_model, 1)

    def actorcritic_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, return_value=True, print_=False):

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        last_hidden_state = outputs.decoder_hidden_states[-1]
        
        if print_:
            print(input_ids, attention_mask)
            print(decoder_input_ids, decoder_attention_mask)
            print(last_hidden_state, outputs['logits'])

        if return_value:
            value = self.value_head(last_hidden_state).squeeze(-1)
            return outputs['logits'], value
        else:
            return outputs['logits']

    def actor_generate(self, input_ids, attention_mask, max_length=100, **kwargs):
        return self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=0,
            top_p=1,
            temperature=1,
            **kwargs
        )



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
                 actor_path: str='',
                 max_seq_length1: int = -1,
                 max_seq_length2: int = -1,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 train_data_dir: str = '',
                 data_dir: str = '',
                 seed: int = 42):

        super().__init__()

        self.actor_path = actor_path
        self.max_seq_length1  = max_seq_length1
        self.max_seq_length2  = max_seq_length2
        self.train_batch_size = train_batch_size
        self.eval_batch_size  = eval_batch_size
        self.data_dir         = data_dir
        self.train_data_dir   = train_data_dir
        self.seed             = seed

        self.tokenizer = AutoTokenizer.from_pretrained(actor_path, use_fast=True)   

    def load_dataset(self):
        train_file_name = self.train_data_dir
        train_examples = [
            example for example in load_json(train_file_name)
            if example['min_con'] > 0.75 and example['avg_con'] > 0.95
        ]

        dev_file_name  = os.path.join(self.data_dir, 'train.json')
        test_file_name = os.path.join(self.data_dir, 'train.json')
        dev_examples   = load_json(dev_file_name)
        test_examples  = load_json(test_file_name) 

        self.index_example_by_tao(train_examples+dev_examples)

        print('--------- data statistic ---------')
        for thre in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
            _examples = [
                example for example in load_json(train_file_name)
                if example['min_con'] > thre
            ]
            print(thre, len(_examples))

        self.raw_datasets = {
            'train': self.make_example(train_examples),
            'dev'  : self.make_example(dev_examples,  repk=2),
            'test' : self.make_example(test_examples, repk=2),
        }

        print('--------- data statistic ---------')
        print('train:', len(self.raw_datasets['train']))
        print('dev:',   len(self.raw_datasets['dev']))
        print('test:',  len(self.raw_datasets['test']))
        print()

    def index_example_by_tao(self, examples):
        self.example_indexed_by_tao = defaultdict(list)

        for example in examples:
            triplets_seq = (example['triplets_seq'] 
                            if 'triplets_seq' in example
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

    def make_example_old(self, examples, repk=1):
        new_examples = []
        for example in examples:
            for i in range(repk):
                new_example = {
                    'ID': example['ID'],
                    'syn_seq' : example['syn_seq'],
                    'syn_sentence': example['sentence'],
                }

                tao = get_triplet_num_aspect_num_opinion_num((
                    example['triplets_seq'] if 'triplets_seq' in example else
                    example['triplets']
                ))
                triplets_seq, triplet_sentence = self.sample_triplets_by_tao(tao)
                new_example['triplet_sentence'] = triplet_sentence
                new_example['triplets_seq'] = triplets_seq
                new_examples.append(new_example)

        return new_examples

    def get_tao(self, example):
        tao = get_triplet_num_aspect_num_opinion_num((
            example['triplets_seq'] if 'triplets_seq' in example else
            example['triplets']
        ))
        return tao

    def make_example(self, examples, repk=1):
        new_examples = []
        for example in examples:
            tao = self.get_tao(example)
            for i in range(repk):
                syn_seq, syn_sentence = self.sample_syn_by_tao(tao)
                triplets_seq = example['triplets_seq'] if 'triplets_seq' in example else make_triplets_seq(example['triplets'])

                new_example = {
                    'ID': f"{example['ID']}-{i}",
                    'syn_seq': syn_seq,
                    'syn_sentence': syn_sentence,
                    'triplets_seq': triplets_seq,
                    'triplet_sentence': example['sentence'],
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

    def __call__(self, examples):
        conditions = []
        real_syn_sentences = []
        real_triplet_sentences = []

        for example in examples:
            syn_seq = example['syn_seq'].lower()
            triplets_seq = self.make_triplets_seq(example)
            condition = triplets_seq + ' ; ' + syn_seq
            conditions.append(condition)

            real_syn_sentences.append(example['syn_sentence'])
            real_triplet_sentences.append(example['triplet_sentence'])

        lengths = self.tok(real_syn_sentences, self.max_seq_length2)['attention_mask'].sum(dim=-1)

        if self.mode in ('dev', 'test'):

            batch_encodings = self.tok(conditions, self.max_seq_length1)
            input_ids = batch_encodings['input_ids']
            attention_mask = batch_encodings['attention_mask']
            str_real_actions = real_triplet_sentences

            return {
                'states': input_ids,
                'states_mask': attention_mask,
                'str_real_actions': str_real_actions,
                'lengths': lengths,
            }

        elif self.mode == 'train':

            return {
                'str_states': conditions,
                'lengths': lengths,
            }

    def make_triplets_seq(self, example):
        if 'triplets_seq' in example:
            return example['triplets_seq']

        triplets_seq = make_triplets_seq(example['triplets'])
        example['triplets_seq'] = triplets_seq
        return triplets_seq




class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module, reward_fn):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.data_module = data_module
        self.tokenizer = data_module.tokenizer
        self.reward_fn = reward_fn

        self.model     = ActorModelWithValueHead.from_pretrained(self.hparams.actor_path)
        self.ref_model = ActorModelWithValueHead.from_pretrained(self.hparams.actor_path)

        self.automatic_optimization = False

        self.running_moments = RunningMoments()
        self.ref_mean = None
        self.ref_std  = None

    def _make_model_dir(self):
        return os.path.join(self.hparams.output_dir, 'model', f'b={self.hparams.output_sub_dir}')

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = self._make_model_dir()
        print(f'## save model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)
        self.tokenizer.save_pretrained(dir_name)

    def load_model(self):
        dir_name = self._make_model_dir()
        print(f'## load model to {dir_name}')
        self.model = ActorModelWithValueHead.from_pretrained(dir_name)

    def setup(self, stage):
        if stage == 'fit':
            # 需要 learn_batch_size 可以被 chunk_size 整除
            effective_batch_size = self.hparams.learn_batch_size
            dataset_size = len(self.data_module.raw_datasets['train'])
            self.estimated_stepping_batches = (dataset_size // effective_batch_size) * self.hparams.max_epochs
            print('self.estimated_stepping_batches', self.estimated_stepping_batches)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.actor_lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.estimated_stepping_batches,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        ppo_rl_elements = self.make_experience(batch)
        self.learn(ppo_rl_elements)

    def make_experience(self, batch):

        def state_collator(data):
            str_states = [line[0] for line in data]
            lengths = [line[1] for line in data]
            batch_encodings = tok(self.tokenizer, str_states, self.hparams.max_seq_length1)

            return {
                'input_ids': batch_encodings['input_ids'],
                'attention_mask': batch_encodings['attention_mask'],
                'lengths': torch.tensor(lengths),
        }

        dataloader = DataLoader(
            dataset=StateLengthDataset(batch['str_states'], batch['lengths']),
            batch_size=self.hparams.chunk_size,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=8,
            collate_fn=state_collator,
            num_workers=1
        )

        self.model.eval()
        ppo_rl_elements = []

        pbar = tqdm(dataloader, desc='make_experience')
        for state_batch in pbar:
            
            state_batch = self.to_cuda(state_batch)
            lengths = state_batch.pop('lengths')
            
            actions = self.model.actor_generate(**state_batch)

            str_actions = self.decode(actions)
            str_states  = self.decode(state_batch['input_ids'])

            ################## get reward ####################
            reward_scores, stat = self.get_rewards(str_states, str_actions, lengths)

            if self.hparams.scale_reward == 'ref':
                if self.ref_mean is None:
                    self.ref_mean = reward_scores.mean()
                    self.ref_std  = reward_scores.std()
                reward_scores = reward_scores / self.ref_std

            elif self.hparams.scale_reward == 'running':
                self.running_moments.update(reward_scores)
                reward_scores = reward_scores / self.running_moments.std

            clip_reward = self.hparams.cliprange_reward
            if clip_reward:
                reward_scores = torch.clip(reward_scores, -clip_reward, clip_reward)

            #################### logits ######################
            with torch.no_grad():
                # old_actions = actions
                # actions, actions_mask = self.retokenize_actions(str_actions)
                # for oa, a in zip(old_actions, actions):
                #     if (oa != a).any():
                #         print(self.tokenizer.convert_ids_to_tokens(oa))
                #         print(self.tokenizer.convert_ids_to_tokens(a))
                #         print()

                actions_mask = self.get_mask(actions)

                logits, values = self.model.actorcritic_forward(
                    **state_batch,
                    decoder_input_ids=actions,
                    decoder_attention_mask=actions_mask
                )

                ref_logits = self.ref_model.actorcritic_forward(
                    **state_batch,
                    decoder_input_ids=actions,
                    decoder_attention_mask=actions_mask,
                    return_value=False
                )

            logprobs = self.get_logprobs(logits[:, :-1, :], actions[:, 1:])
            ref_logprobs = self.get_logprobs(ref_logits[:, :-1, :], actions[:, 1:])

            log_ratio = (logprobs - ref_logprobs) * actions_mask[:, :-1]
            kl_penalty = self.hparams.beta * -log_ratio.cpu()
            mean_kl = masked_mean(log_ratio.exp() - 1 - log_ratio, actions_mask[:, 1:], dim=1)

            cliprange_kl = self.hparams.cliprange_kl
            if cliprange_kl:
                kl_penalty = torch.clip(kl_penalty, -cliprange_kl, cliprange_kl)
            ################## to cpu #######################
            logprobs = logprobs.cpu()
            values   = values.cpu()[:, :-1]
            actions  = actions.cpu()

            ends = actions_mask.sum(1)-1
            values = [values[ix, :ends[ix]] for ix in range(len(str_states))]
            logprobs = [logprobs[ix, :ends[ix]] for ix in range(len(str_states))]
            kl_penalty = [xs[:ends[ix]] for ix, xs in enumerate(kl_penalty)]

            actions = [actions[ix, :ends[ix]+1] for ix in range(len(str_states))]
            actions_mask = [actions_mask[ix, :ends[ix]+1] for ix in range(len(str_states))]

            for sample_idx in range(len(str_states)):
                reward = kl_penalty[sample_idx]
                reward[-1] += reward_scores[sample_idx].cpu()

                ppo_rl_elements.append(
                    PPORLELement(
                        state=str_states[sample_idx],
                        action=actions[sample_idx],
                        action_mask=actions_mask[sample_idx],
                        logprob=logprobs[sample_idx],
                        value=values[sample_idx],
                        reward=reward,
                    )
                )

            alignment_score = stat['alignment_scores'].mean().item()
            fluency_score   = stat['fluency_scores'].mean().item()

            pbar.set_postfix({
                'fluency': fluency_score,
                'align': alignment_score,
                'mean_kl': mean_kl.mean().item()
            })

        return ppo_rl_elements

    def learn(self, ppo_rl_elements):

        def experience_collator(data):
            states, actions, actions_mask, logprobs, values, rewards = [], [], [], [], [], []
            for state, action, action_mask, logprob, value, reward in data:
                states.append(state)
                actions.append(action)
                actions_mask.append(action_mask)
                logprobs.append(logprob)
                values.append(value)
                rewards.append(reward)

            state_batch_encodings = tok(self.tokenizer, states, self.hparams.max_seq_length1)
            states = state_batch_encodings['input_ids']
            states_mask = state_batch_encodings['attention_mask']

            actions = pad_sequence(actions, padding_value=0., batch_first=True)
            actions_mask = pad_sequence(actions_mask, padding_value=0., batch_first=True)

            logprobs = pad_sequence(logprobs, padding_value=0., batch_first=True)
            values   = pad_sequence(values,   padding_value=0., batch_first=True)
            rewards  = pad_sequence(rewards,  padding_value=0., batch_first=True)

            return {
                'states': states.cuda(),
                'states_mask': states_mask.cuda(),
                'actions': actions.cuda(),
                'actions_mask': actions_mask.cuda(),
                'logprobs': logprobs.cuda(),
                'values': values.cuda(),
                'rewards': rewards.cuda(),
            }

        dataloader = DataLoader(
            ExperienceDataset(ppo_rl_elements),
            batch_size=self.hparams.learn_batch_size,
            shuffle=True,
            collate_fn=experience_collator
        )

        actor_optim = self.optimizers()
        scheduler   = self.lr_schedulers()
        self.model.train()

        pbar = tqdm(dataloader, desc='learn')
        for batch in pbar:

            old_values   = batch['values']
            old_rewards  = batch['rewards']
            old_logprobs = batch['logprobs']

            actions = batch['actions']
            actions_mask = batch['actions_mask']

            states = batch['states']
            states_mask = batch['states_mask']

            ps, vs = [], [] # policy_loss, value_loss
            for _ in range(self.hparams.n_updates_per_batch):

                advantages, returns = self.get_advantages_and_returns(old_values, old_rewards)
                logits, values_pred = self.model.actorcritic_forward(
                    input_ids=states,
                    attention_mask=states_mask,
                    decoder_input_ids=actions,
                    decoder_attention_mask=actions_mask,
                )

                logprobs = self.get_logprobs(logits[:, :-1, :], actions[:, 1:])
                
                loss, stats = self.compute_loss(
                    logprobs=logprobs,
                    values=values_pred,
                    old_logprobs=old_logprobs,
                    old_values=old_values,
                    advantages=advantages,
                    returns=returns,
                    actions_mask=actions_mask,
                )

                if torch.isnan(loss):
                    print(stats)
                    print(logprobs.size(), old_logprobs.size())
                    print(logprobs)
                    print(old_logprobs)
                    print(values_pred)
                    print(actions)
                    print(actions_mask)

                    raise ValueError('Loss is nan')

                p, v = stats[:2]
                ps.append(p)
                vs.append(v)

                actor_optim.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(actor_optim, gradient_clip_val=self.hparams.gradient_clip_val_manual)
                actor_optim.step()
                scheduler.step()

            p = self.get_mean(ps)
            v = self.get_mean(vs)
            pbar.set_postfix({'p': p, 'v': v})

    def get_advantages_and_returns(self, values, rewards):
        length = values.size(1)
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(length)):
            nextvalues = values[:, t+1] if t < length -1 else 0.
            delta = rewards[:, t] + self.hparams.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.hparams.gamma * self.hparams.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = whiten(advantages)
        return advantages.detach(), returns

    def compute_loss(self, logprobs, values, old_logprobs, old_values, advantages, returns, actions_mask):

        assert logprobs.shape[1] == old_logprobs.shape[1]
        response_length = min(logprobs.shape[1], old_logprobs.shape[1])

        logprobs = logprobs[:, :response_length]
        values = values[:, :response_length]
        actions_mask = actions_mask[:, :response_length]

        old_logprobs = old_logprobs[:, :response_length]
        old_values = old_values[:, :response_length]
        advantages = advantages[:, :response_length]
        returns = returns[:, :response_length]

        values_clipped = torch.clamp(
            values,
            old_values - self.hparams.cliprange_value,
            old_values + self.hparams.cliprange_value,
        )
        
        n = actions_mask.sum()

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * actions_mask) / n

        log_ratio = (logprobs - old_logprobs) * actions_mask
        ratio = torch.exp(log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.hparams.cliprange,
            1.0 + self.hparams.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * actions_mask) / n 

        loss = pg_loss + self.hparams.vf_coef * vf_loss
        return loss, [pg_loss.item(), vf_loss.item(), advantages, logprobs, old_logprobs]

    def to_cuda(self, batch):
        if type(batch) is dict:
            return {k: v.cuda() for k, v in batch.items()}
        else:
            raise NotImplementedError

    def get_mask(self, input_ids):
        eos_token_id = self.model.config.eos_token_id
        pad_token_id = self.model.config.pad_token_id

        eos_flag = (input_ids == eos_token_id)
        eos_flag = torch.cat([eos_flag[:, :1], eos_flag[:, :-1]], dim=1)
        attention_mask = torch.cumsum(eos_flag, dim=1)
        attention_mask = (attention_mask == 0).bool()
        input_ids[~attention_mask] = pad_token_id
        # input_ids[:, -1] = eos_token_id

        return attention_mask.long()

    def retokenize_actions(self, tensor_or_string):
        if type(tensor_or_string[0]) != str:
            strings = self.decode(tensor_or_string)
        else:
            strings = tensor_or_string

        batch_encodings = tok(self.tokenizer, strings, self.hparams.max_seq_length2)  
        actions = batch_encodings['input_ids'].cuda()
        actions_mask = batch_encodings['attention_mask'].cuda()

        return self.add_decoder_start(actions, actions_mask)

    def add_decoder_start(self, input_ids, attention_mask):
        decoder_start_token_id = self.model.config.decoder_start_token_id
        decoder_start_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id, device=input_ids.device)
        input_ids = torch.cat([decoder_start_ids, input_ids], dim=1)
        attention_mask = torch.cat([attention_mask[:, :1], attention_mask], dim=1)
        assert input_ids.size() == attention_mask.size(), f'{input_ids.size()} != {attention_mask.size()}'
        return input_ids, attention_mask

    def get_logprobs(self, logits, actions):
        return logprobs_of_labels(logits, actions)

    def decode(self, input_ids):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    
    def get_mean(self, lst):
        return sum(lst) / len(lst) if len(lst) > 0 else 0

    def get_rewards(self, *args, **kwargs):
        fluency_scores, alignment_scores, length_penalty = self.reward_fn(*args, **kwargs)

        reward = (
            fluency_scores * self.hparams.fluency_coef 
            + alignment_scores * self.hparams.alignment_coef
            - length_penalty * self.hparams.length_coef 
        )

        return reward, {
           'fluency_scores': fluency_scores, 
           'alignment_scores': alignment_scores, 
           'length_penalty': length_penalty, 
        }

    def validation_step(self, batch, batch_idx, model=None):
        states = batch['states']
        states_mask = batch['states_mask']
        lengths = batch['lengths']
        str_real_actions = batch['str_real_actions']

        model = self.model if model is None else model

        actions = model.actor_generate(
            input_ids=states, 
            attention_mask=states_mask,
        )

        str_actions = self.decode(actions)
        str_states  = self.decode(states)
        # actions, actions_mask = self.retokenize_actions(str_actions)
        actions_mask = self.get_mask(actions)

        reward_scores, stat = self.get_rewards(str_states, str_actions, lengths)        
        real_stat = self.get_rewards(str_states, str_real_actions, lengths)[1]

        flu_accu = (stat['fluency_scores'] > 0).float().mean()
        ali_accu = (stat['alignment_scores'] > 0).float().mean()

        fake_length = actions_mask.sum(dim=-1).float().mean()
        real_length = lengths.float().mean()

        with torch.no_grad():
            logits = self.model.actorcritic_forward(
                input_ids=states, 
                attention_mask=states_mask,
                decoder_input_ids=actions,
                decoder_attention_mask=actions_mask,
                return_value=False
            )

            ref_logits = self.ref_model.actorcritic_forward(
                input_ids=states,
                attention_mask=states_mask,
                decoder_input_ids=actions,
                decoder_attention_mask=actions_mask,
                return_value=False,
            )

        logprobs = self.get_logprobs(logits[:, :-1, :], actions[:, 1:])
        ref_logprobs = self.get_logprobs(ref_logits[:, :-1, :], actions[:, 1:])

        log_ratio = (logprobs - ref_logprobs) * actions_mask[:, 1:]
        mean_kl = masked_mean(log_ratio.exp() - 1 - log_ratio, actions_mask[:, 1:], dim=1)

        return {
            'conditions': str_states,
            'fake_sentences': str_actions,
            'actions': actions,
            'real_sentences': str_real_actions,
            
            'fluency_scores'  : stat['fluency_scores'],
            'alignment_scores': stat['alignment_scores'],
            'length_penalty'  : stat['length_penalty'],
            
            'real_fluency_scores'  : real_stat['fluency_scores'],
            'real_alignment_scores': real_stat['alignment_scores'],

            'flu_accu': flu_accu.item(),
            'ali_accu': ali_accu.item(),

            'fake_length': fake_length.item(),
            'real_length': real_length.item(),

            'mean_kl': mean_kl,
            'reward_scores': reward_scores,
            'd_mean_kl': log_ratio.exp() - 1 - log_ratio,
        }

    def eval_epoch_end(self, outputs):
        flu_accu = self.get_mean([output['flu_accu'] for output in outputs])
        ali_accu = self.get_mean([output['ali_accu'] for output in outputs])

        fake_length = self.get_mean([output['fake_length'] for output in outputs])
        real_length = self.get_mean([output['real_length'] for output in outputs])

        mean_kl = self.get_mean([output['mean_kl'].mean() for output in outputs])

        fluency_scores = self.get_mean([output['fluency_scores'].mean() for output in outputs])
        alignment_scores = self.get_mean([output['alignment_scores'].mean() for output in outputs])
        length_penalty = self.get_mean([output['length_penalty'].mean() for output in outputs])

        reward_scores = self.get_mean([output['reward_scores'].mean() for output in outputs])

        real_fluent_scores = self.get_mean([output['real_fluency_scores'].mean() for output in outputs])
        real_alignment_scores = self.get_mean([output['real_alignment_scores'].mean() for output in outputs])

        eval_metric = {
            'f_accu' : flu_accu,
            'flu': fluency_scores,
            'real_flu': real_fluent_scores,

            'a_accu' : ali_accu,
            'ali': alignment_scores,
            'real_ali': real_alignment_scores,
            
            'l_penalty': length_penalty,
            'fake_len': fake_length,
            'real_len': real_length,

            'mean_kl': mean_kl,
            'monitor': reward_scores,
        }
        return eval_metric

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
        output_ref = self.validation_step(batch, batch_idx, model=self.ref_model)

        # print(output['mean_kl'])
        # print()
        for condition, ofsent, fsent, action, rsent, ofscore, fscore, rscore, oali, ali, rali, kl, okl, dkl in zip(
            
            output['conditions'],
            output_ref['fake_sentences'],
            output['fake_sentences'],
            output['actions'],
            output['real_sentences'],
            
            output_ref['fluency_scores'],
            output['fluency_scores'],
            output['real_fluency_scores'],

            output_ref['alignment_scores'],
            output['alignment_scores'],
            output['real_alignment_scores'],

            output['mean_kl'],
            output_ref['mean_kl'],

            output['d_mean_kl'],
        ):  
            if random.random() < .02:
            # if kl > 10000:
                print(condition)
                print('old: ', ofsent, ofscore.item(), oali.item(), okl.item())
                print('fake:', fsent,  fscore.item(),  ali.item(), kl.item())
                # print([(f'{action_token}({dkl_token:.4f})') for action_token, dkl_token in zip(self.tokenizer.convert_ids_to_tokens(action[1:]), dkl)])

                print('real:', rsent,  rscore.item(),  rali.item())
                print()

        return output_ref, output

    def test_epoch_end(self, outputs):
        outputs_1 = [output[0] for output in outputs]
        outputs_2 = [output[1] for output in outputs]

        self.test_metric_1 = self.eval_epoch_end(outputs_1)
        self.test_metric_2 = self.eval_epoch_end(outputs_2)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--actor_lr", default=1e-5, type=float)
        
        parser.add_argument("--beta", default=0.01, type=float)  # kl
        parser.add_argument("--lam", default=0.95, type=float)   # 
        parser.add_argument("--gamma", default=0.99, type=float) # 

        parser.add_argument("--fluency_coef", default=1., type=float)
        parser.add_argument("--alignment_coef", default=1., type=float)
        parser.add_argument("--length_coef", default=1., type=float)
        parser.add_argument("--vf_coef", default=0.5, type=float)

        parser.add_argument("--learn_batch_size", default=20, type=int)
        parser.add_argument("--chunk_size", default=192, type=int)
        parser.add_argument("--n_updates_per_batch", default=4, type=int)

        parser.add_argument("--cliprange", default=0.2, type=float)
        parser.add_argument("--cliprange_value", default=0.2, type=float)
        parser.add_argument("--cliprange_reward", default=3., type=float)
        parser.add_argument("--cliprange_kl", default=0., type=float)
        parser.add_argument("--scale_reward", type=str, default=None)

        parser.add_argument("--eps", default=1e-5, type=float)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--gradient_clip_val_manual", default=1, type=float)
        parser.add_argument("--max_timesteps", default=100, type=int)
        parser.add_argument("--warmup_steps", default=0, type=int)

        parser.add_argument("--fluency_model_path", type=str)
        parser.add_argument("--alignment_model_path", type=str)
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--output_sub_dir", type=str)
        parser.add_argument("--do_train", action='store_true')

        return parser



PPORLELement = namedtuple(
    "PPORLELement",
    [
        "state",
        "action",
        "action_mask",
        "logprob",
        "value",
        "reward",
    ]
)


class ExperienceDataset(Dataset):
    def __init__(self, ppo_rl_elements):
        self.data = ppo_rl_elements

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = (
            self.data[idx].state,
            self.data[idx].action,
            self.data[idx].action_mask,
            self.data[idx].logprob,
            self.data[idx].value,
            self.data[idx].reward,
        )
        return item



class StateLengthDataset(Dataset):
    def __init__(self, states, lengths):
        self.states  = states
        self.lengths = lengths.tolist()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        item = (
            self.states[idx],
            self.lengths[idx],
        )
        return item



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
        self.print_dict('[old] ', pl_module.test_metric_1)
        self.print_dict('[test]', pl_module.test_metric_2)
        print()



class Reward_Fn:
    def __init__(self, args):
        self.args = args
        self.fluency_model = T5ForClassification.from_pretrained(args.fluency_model_path)
        self.fluency_model.cuda()
        self.fluency_model.eval()

        self.alignment_model = T5ForClassification.from_pretrained(args.alignment_model_path)
        self.alignment_model.cuda()
        self.alignment_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(args.fluency_model_path)

    def get_triplets_from_states(self, states):
        return_triplets_seqs = []
        for state in states:
            try:
                triplets_seq = state[:state.index(' ; root')]
            except:
                index = -state[::-1].index(';')-1
                syn_seq = state[index+1:].strip()
                triplets_seq = state[:index].strip()

            return_triplets_seqs.append(triplets_seq)
        return return_triplets_seqs

    def __call__(self, states, actions, lengths):
        
        triplets_seqs = self.get_triplets_from_states(states)

        with torch.no_grad():
            abatch_encodings = tok(self.tokenizer, actions, self.args.max_seq_length2)
            fluency_scores  = self.fluency_model(
                input_ids=abatch_encodings['input_ids'].cuda(), 
                attention_mask=abatch_encodings['attention_mask'].cuda(),
            )

            triplets_actions = [triplets_seq + ' ; ' + action for triplets_seq, action in zip(triplets_seqs, actions)]
            tbatch_encodings = tok(self.tokenizer, triplets_actions, self.args.max_seq_length1)
            alignment_scores = self.alignment_model(
                input_ids=tbatch_encodings['input_ids'].cuda(), 
                attention_mask=tbatch_encodings['attention_mask'].cuda(),
            )

            alengths = abatch_encodings['attention_mask'].sum(dim=-1).cuda()

            length_penalty = (lengths - alengths)
            length_penalty = (length_penalty > 0) * length_penalty # 只计算小于预期长度的句子的损失
            length_penalty = length_penalty ** 2 / lengths

        return fluency_scores, alignment_scores, length_penalty



def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LightningModule.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    data_module = DataModule.from_argparse_args(args)
    data_module.load_dataset()

    reward_fn = Reward_Fn(args)
    model = LightningModule(args, data_module, reward_fn)

    logging_callback = LoggingCallback()  
    kwargs = {
        'callbacks': [logging_callback],
        'logger': False,
        'enable_checkpointing': False,
        'num_sanity_val_steps': 100 if args.do_train else 0,
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
    torch.set_printoptions(linewidth=300)
    main()