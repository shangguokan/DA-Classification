import re
import nltk
from collections import defaultdict, OrderedDict
from dataset.swda.swda import CorpusReader

def load_swda_corpus(conversation_list, concatenate_interruption):
    corpus = defaultdict(lambda: defaultdict(list))
    vocabulary = OrderedDict({'<TOKEN_PAD>': 0, '<PRE_CONTEXT_PAD>': 1, '<POST_CONTEXT_PAD>': 2, '<UNK>': 3, '<-->': 4})
    tag_set = set()
    speaker_set = set()
    tokenizer = nltk.tokenize.TweetTokenizer()

    for trans in CorpusReader('dataset/swda/swda').iter_transcripts():
        conversation_id = 'sw' + str(trans.conversation_no)
        if conversation_id not in conversation_list:
            continue

        for utt in trans.utterances:
            words = ' '.join(utt.text_words(filter_disfluency=True))
            # print(words)
            words = re.sub(r'<<[^>]*>>', '', words)
            words = re.sub(r'\*.*$', '', words)
            words = re.sub(r'[#)(]', '', words)
            words = re.sub(r'--', '', words)
            words = re.sub(r' -$', '', words)
            while True:
                output = re.sub(r'([A-Z]) ([A-Z])\b', '\\1\\2', words)
                if output == words:
                    break
                words = output
            # print(words)
            words = tokenizer.tokenize(words)
            words = ' '.join(words)
            words = re.sub(r' -s', 's', words)
            words = re.sub(r' -', '-', words)
            # print(words)

            words = words.lower()

            words = words.split()
            if len(words) == 0:
                words = ['<UNK>'] + words
            if len(words) == 1 and words[0] in '.,?':
                words = ['<UNK>'] + words

            # words to indexes
            w2i = []
            for word in words:
                if word not in vocabulary.keys():
                    word_idx = len(vocabulary.keys())
                    vocabulary[word] = word_idx
                    w2i.append(word_idx)
                else:
                    w2i.append(vocabulary[word])

            utt_tag = utt.damsl_act_tag()
            if (utt_tag != '+') or (utt_tag == '+' and concatenate_interruption is False):
                corpus[conversation_id]['sequence'].append(w2i)
                corpus[conversation_id]['text'].append(words)
                corpus[conversation_id]['tag'].append(utt_tag)
                corpus[conversation_id]['speaker'].append(utt.caller)
                tag_set.add(utt_tag)
                speaker_set.add(utt.caller)
            # resolve issue related to tag '+' (check section 2 of the Coders' Manual)
            else:
                concatenated = False
                for i in reversed(range(len(corpus[conversation_id]['sequence']))):
                    if corpus[conversation_id]['speaker'][i] == utt.caller:
                        corpus[conversation_id]['sequence'][i].extend([vocabulary['<-->']]+w2i)
                        corpus[conversation_id]['text'][i].extend(['<-->']+words)
                        concatenated = True
                        break
                assert concatenated is True, 'please comment out the line #195 of swda/swda.py.'

    return corpus, vocabulary, tag_set, speaker_set
