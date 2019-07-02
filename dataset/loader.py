import re
from collections import defaultdict
from dataset.swda.swda import CorpusReader


def tokenize_corpus(corpus, tokenizer):
    for cid in corpus.keys():
        tokenized_sentences = []
        sequences = []
        for sentence in corpus[cid]['text']:
            tokenized_sentences.append(
                tokenizer.encode_as_pieces(' '.join(sentence))
            )
            print('|'.join(tokenizer.encode_as_pieces(' '.join(sentence))))
            sequences.append(
                tokenizer.encode_as_ids(' '.join(sentence))
            )
        corpus[cid]['tokenized_text'] = tokenized_sentences
        corpus[cid]['sequence'] = sequences

    return corpus


def load_swda_corpus(conversation_list, concatenate_interruption, do_lowercase):
    corpus = defaultdict(lambda: defaultdict(list))
    tag_set = set()
    speaker_set = set()

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
            words = re.sub(r' -s', 's', words)
            # print(words)

            if do_lowercase is True:
                words = words.lower()

            words = words.split()
            if len(words) == 0:
                words = ['<UNK>'] + words
            if len(words) == 1 and words[0] in '.,?':
                words = ['<UNK>'] + words

            utt_tag = utt.damsl_act_tag()
            if (utt_tag != '+') or (utt_tag == '+' and concatenate_interruption is False):
                corpus[conversation_id]['text'].append(words)
                corpus[conversation_id]['tag'].append(utt_tag)
                corpus[conversation_id]['speaker'].append(utt.caller)
                tag_set.add(utt_tag)
                speaker_set.add(utt.caller)
            # resolve issue related to tag '+' (check section 2 of the Coders' Manual)
            else:
                concatenated = False
                for i in reversed(range(len(corpus[conversation_id]['tag']))):
                    if corpus[conversation_id]['speaker'][i] == utt.caller:
                        corpus[conversation_id]['text'][i].extend(['<CONNECTOR>']+words)
                        concatenated = True
                        break
                assert concatenated is True, 'please comment out the line #195 of swda/swda.py.'

    return corpus, tag_set, speaker_set
