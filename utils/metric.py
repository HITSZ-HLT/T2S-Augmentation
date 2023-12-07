import sacrebleu
from rouge import Rouge
from nltk.tokenize import RegexpTokenizer
from nltk.translate.meteor_score import meteor_score
from typing import List, Tuple, Dict
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch.nn.functional as F

from . import tgenerate_batch, generate_batch



def compute_bleu(references: List[str], candidates: List[str]) -> List:
    '''
    @param reference: 一系列待参考的句子，每个句子和sentence中相应位置的句子对应。
    @param sentences: 一系列新生成的（带评判）的句子
    @return: list of (bleu1, bleu2, bleu3, bleu4)
    '''
    scores = []
    for r, c in zip(references, candidates):
        bleu_score = sacrebleu.corpus_bleu([c], [[r]]).precisions
        scores.append(bleu_score)

    return scores 



def compute_rouge(references: List[str], candidates: List[str]) -> Dict:
    '''
    @param reference: 一系列待参考的句子，每个句子和sentence中相应位置的句子对应。
    @param sentences: 一系列新生成的（带评判）的句子
    @return: list of dict{'rouge-1': xx, 'rouge-2': xx, 'rouge-l':xx}
    '''
    r = Rouge()
    # scores = []
    # for r, c in zip(references, candidates):
    #     r_s = r.get_scores(hyps=[c], refs=[r])
    rouge_score = r.get_scores(hyps=candidates, refs=references, avg=False)
    return rouge_score



def compute_meteor(references: List[str], candidates: List[str]) -> List:
    tokenizer = RegexpTokenizer(r'\w+')
    scores = []
    for r, c in zip(references, candidates):
        t_r = tokenizer.tokenize(r)
        t_c = tokenizer.tokenize(c)
        scores.append(meteor_score([t_r], t_c))
    return scores    



def _distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)



def compute_bleurt(candidates:List[str], references: List[str], batch_size: int) -> List:
    '''
    @param candidates: 一个batch的句子/一个句子/一组句子
    @param references, 和candidates对应的参考句
    @return: list of scores of each sentence
    '''
    # checkpoint = "data/bleurt/test_checkpoint"
    # checkpoint = "utils/_bleurt/test_checkpoint"
    scorer = bleurt_score.BleurtScorer('BLEURT-20-D12')
    # scorer = bleurt_score.LengthBatchingBleurtScorer()
    scores = scorer.score(references=references, candidates=candidates, batch_size=batch_size)
    assert isinstance(scores, list)
    return scores



class FluencyScorer:
    def __init__(self, model_name_or_path='prithivida/parrot_fluency_model'):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.cuda()

    def compute_flu(self, candidates, bz=16, progress_bar=True):
        scores = []

        iterator = (tgenerate_batch(candidates, bz=bz) 
                    if progress_bar else generate_batch(candidates, bz=bz))

        for text in iterator:
            
            kwargs = {
                'padding': 'max_length',
                'return_tensors' : 'pt',
                'truncation' : True,
                'max_length' : 128,
                'text' : ['Sentence: ' + t for t in text],
            }
            
            encodings = self.tokenizer(**kwargs).to('cuda')
            self.model.eval()
            predictions = self.model(**encodings)[0]
            predictions = F.softmax(predictions, dim=-1)

            # LABEL_0 = Bad Fluency, LABEL_1 = Good Fluency
            for score in predictions[:, 1].detach().cpu().numpy():
                scores.append(score)

        return scores



class F1_Measure:
    def __init__(self):
        self.pred_list = []
        self.true_list = []

    def pred_inc(self, idx, preds):
        for pred in preds:
            self.pred_list.append((idx, pred))
            
    def true_inc(self, idx, trues):
        for true in trues:
            self.true_list.append((idx, true))
            
    def report(self):
        self.f1, self.p, self.r = self.cal_f1(self.pred_list, self.true_list)
        return self.f1
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise NotImplementedError

    def cal_f1(self, pred_list, true_list):
        n_tp = 0
        for pred in pred_list:
            if pred in true_list:
                n_tp += 1    
        _p = n_tp / len(pred_list) if pred_list else 1
    
        n_tp = 0
        for true in true_list:
            if true in pred_list:
                n_tp += 1 
        _r = n_tp / len(true_list) if true_list else 1

        f1 = 2 * _p * _r / (_p + _r) if _p + _r else 0

        return f1, _p, _r