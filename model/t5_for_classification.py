import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from einops.layers.torch import Rearrange

from transformers import T5EncoderModel



class T5ForClassification(T5EncoderModel):
    def __init__(self, config):
        super().__init__(config)

        if not hasattr(config, 'd_head'):
            config.d_head = config.d_model

        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_head),
            nn.ReLU(),
            nn.Linear(config.d_head, 1),
            Rearrange('... 1 -> ...'),
        )

    def forward(self, input_ids, attention_mask):
        encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # max-pooling
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = last_hidden_state - (1-attention_mask)[:, :, None] * 10000.
        hidden = torch.max(last_hidden_state, dim=1).values

        rewards = self.head(hidden)
        return rewards

    def forward_train(self, fake_input_ids, real_input_ids, fake_attention_mask, real_attention_mask):

        fake_score = self.forward(fake_input_ids, fake_attention_mask)
        real_score = self.forward(real_input_ids, real_attention_mask)

        loss1 = F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
        loss2 = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
        loss  = loss1 + loss2
        
        accu1 = (real_score>0).float().mean()
        accu2 = (fake_score<0).float().mean()
        accu  = (accu1+accu2)/2

        return {
            'loss' : loss,
            'accu1': accu1,
            'accu2': accu2,
            'accu' : accu,
            'real_sentences': real_input_ids,
            'fake_sentences': fake_input_ids,
            'fake_score': fake_score,
            'real_score': real_score
        }
