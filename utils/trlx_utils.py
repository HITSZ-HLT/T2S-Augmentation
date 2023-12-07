import torch
import torch.nn.functional as F
from typing import Any, Dict, Iterable, Tuple



class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """Updates running moments from batch's moments computed across ranks"""
        # if dist.is_initialized():
        #     xs_mean, xs_var, xs_count = get_global_statistics(xs)
        # else:
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True) -> torch.Tensor:
    """Whitens values"""
    var, mean = torch.var_mean(xs)
    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels
    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


# def tok(tokenizer, text, max_seq_length):
#     kwargs = {
#         'text': text,
#         'return_tensors': 'pt'
#     }

#     if max_seq_length in (-1, 'longest'):
#         kwargs['padding'] = True

#     else:
#         kwargs['max_length'] = max_seq_length
#         kwargs['padding'] = 'max_length'
#         kwargs['truncation'] = True

#     batch_encodings = tokenizer(**kwargs)
#     return batch_encodings


def tok(tokenizer, text, max_seq_length):
    kwargs = {
        'text': text,
        'return_tensors': 'pt'
    }

    kwargs['padding'] = True
    kwargs['truncation'] = True
    kwargs['max_length'] = max_seq_length

    batch_encodings = tokenizer(**kwargs)
    return batch_encodings


    
def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, keepdim: bool = False) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim, keepdim=keepdim)
    mask_sum = mask.sum(dim=dim, keepdim=keepdim)
    mean = tensor / (mask_sum + 1e-8)
    return mean

