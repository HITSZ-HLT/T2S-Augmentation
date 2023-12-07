import random


_sentiment_to_word = {
    'POS': 'positive',
    'NEU': 'neutral' ,
    'NEG': 'negative',
    'positive': 'POS',
    'neutral' : 'NEU',
    'negative': 'NEG',
}
def sentiment_to_word(key):
    if key not in _sentiment_to_word:
        return 'UNK'
    return _sentiment_to_word[key]



def make_triplets_seq(triplets):
    if type(triplets[0]) is not dict:
        triplets = [
            {
                'aspect': triplet[0],
                'opinion': triplet[1],
                'sentiment': triplet[2],
            }
            for triplet in triplets
        ]

    triplets_seq = []
    for triplet in sorted(
        triplets,
        key=lambda t: (t['aspect'][0], t['opinion'][0])
    ):  
        triplet_seq = (
            triplet['aspect'][-1] + 
            ' | ' + 
            triplet['opinion'][-1] + 
            ' | ' + 
            sentiment_to_word(triplet['sentiment'])
        )
        triplets_seq.append(triplet_seq)

    return ' ; '.join(triplets_seq)




def parse_triplet_seq(triplet_seq, sentence=None):
    if triplet_seq.count('|') != 2:
        return False

    aspect, opinion, sentiment = triplet_seq.split('|')
    aspect  = aspect.strip()
    opinion = opinion.strip()
    sentiment = sentiment_to_word(sentiment.strip())
    if sentiment == 'UNK':
        return False

    if sentence is not None:
        if aspect not in sentence or len(aspect) == 0:
            return False

        if opinion not in sentence or len(opinion) == 0:
            return False 

    return aspect, opinion, sentiment



def parse_triplets_seq(triplets_seq, sentence=None, if_not_complete_drop_all=False):
    triplets = []
    for triplet_seq in triplets_seq.split(';'):
        parse_result = parse_triplet_seq(triplet_seq.strip(), sentence)
        if not parse_result:
            if if_not_complete_drop_all:
                return False
        else:
            triplets.append(parse_result)

    return triplets







def find_nearset_span(sentence, span, ref_start, ref_end):
    ref_loc = (ref_start +  ref_end-1)/2

    candidates = []
    for i in range(len(sentence)):
        if sentence[i: i+len(span)] == span:
            candidates.append((abs((i+i+len(span)-1)/2 - ref_loc), i, i+len(span)))

    _, s, e = min(candidates)
    return s, e


def find_nearset_span_pair(sentence, aspect, opinion):
    candidates = []
    for i in range(len(sentence)):
        if sentence[i: i+len(aspect)] == aspect:
            astart, aend = i, i+len(aspect)
            ostart, oend = find_nearset_span(sentence, opinion, astart, aend)
            candidates.append((abs(astart+aend-ostart-oend), astart, aend, ostart, oend))

    _, astart, aend, ostart, oend = min(candidates)
    return astart, aend, ostart, oend



def parse_triplets_seq_char(triplets_seq, sentence):
    
    triplets = parse_triplets_seq(triplets_seq, sentence, if_not_complete_drop_all=True)

    if triplets is False or len(triplets) == 0:
        return False
    
    char_triplets  = [] 
    for aspect, opinion, sentiment in triplets:
        if sentence.count(aspect) > 1 and sentence.count(opinion) > 1:
            astart, aend, ostart, oend = find_nearset_span_pair(sentence, aspect, opinion)
        
        elif sentence.count(aspect) > 1 and sentence.count(opinion) == 1:

            ostart = sentence.index(opinion)
            oend   = ostart + len(opinion)

            astart, aend = find_nearset_span(sentence, aspect, ostart, oend)

        elif sentence.count(aspect) == 1 and sentence.count(opinion) > 1:

            astart = sentence.index(aspect)
            aend   = astart + len(aspect)

            ostart, oend = find_nearset_span(sentence, opinion, astart, aend)

        else:

            astart = sentence.index(aspect)
            aend   = astart + len(aspect)

            ostart = sentence.index(opinion)
            oend   = ostart + len(opinion)

        # astart = encoding.char_to_token(astart)
        # aend   = encoding.char_to_token(aend-1)+1

        # ostart = encoding.char_to_token(ostart)
        # oend   = encoding.char_to_token(oend-1)+1

        char_triplets.append({
            'aspect': [astart, aend, aspect],
            'opinion': [ostart, oend, opinion],
            'sentiment': sentiment,
        })

    return char_triplets



def parse_triplets_seq_tokens(triplets_seq, sentence, tokenizer):
    char_triplets = parse_triplets_seq_char(triplets_seq, sentence)
    if char_triplets is False:
        return False

    encoding = tokenizer(sentence, add_special_tokens=False)
    tokens   = tokenizer.tokenize(sentence)

    triplets_token = []
    for triplet in char_triplets:
        aspect = triplet['aspect']
        opinion = triplet['opinion']
        sentiment = triplet['sentiment']
        triplet_token = triplet_char_to_token(aspect, opinion, sentiment, encoding, tokens, sentence)
        triplets_token.append(triplet_token)

    return triplets_token



def triplet_char_to_token(aspect, opinion, sentiment, encoding, tokens, sentence):

    def entity_convert(entity):
        char_start, char_end = entity[:2]

        token_start = encoding.char_to_token(char_start)
        token_end   = encoding.char_to_token(char_end-1)+1

        # print(start, end, char_start, char_end, token_start, token_end)
        if token_start is None:
            print(aspect, opinion, sentiment)
            print(sentence, entity)
            print(token_start)
            print(token_end)
            print(tokens)

        assert None not in (token_start, token_end)

        return [
            token_start,
            token_end,
            tokens[token_start:token_end],
            sentence[char_start:char_end]
        ]

    return {
        'aspect'   : entity_convert(aspect),
        'opinion'  : entity_convert(opinion),
        'sentiment': sentiment
    }



def triplet_token_to_char(aspect, opinion, sentiment, encoding):

    def entity_convert(entity):
        token_start, token_end = entity[:2]

        char_start = encoding.token_to_chars(token_start).start
        char_end   = encoding.token_to_chars(token_end-1).end

        assert None not in (char_start, char_end)

        return [
            char_start,
            char_end,
        ]

    return {
        'aspect'   : entity_convert(aspect),
        'opinion'  : entity_convert(opinion),
        'sentiment': sentiment
    }
