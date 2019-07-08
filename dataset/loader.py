import re
import string
from collections import defaultdict
from dataset.swda.swda import CorpusReader


def tokenize_corpus(corpus, tokenizer):
    for cid in corpus.keys():
        tokenized_sentences = []
        sequences = []
        for sentence in corpus[cid]['text']:
            print(sentence)
            tokenized_sentences.append(
                tokenizer.encode_as_pieces(sentence)
            )
            print(tokenizer.encode_as_pieces(sentence))
            sequences.append(
                tokenizer.encode_as_ids(sentence)
            )
            print(tokenizer.encode_as_ids(sentence))
        corpus[cid]['tokenized_text'] = tokenized_sentences
        corpus[cid]['sequence'] = sequences

    return corpus


def load_swda_corpus(conversation_list, concatenate_interruption, do_lowercase, strip_punctuation, do_pretokenization):
    corpus = defaultdict(lambda: defaultdict(list))
    tag_set = set()
    speaker_set = set()
    user_defined_symbols = set()
    user_defined_symbols.add('<CONNECTOR>')

    for trans in CorpusReader('dataset/swda/swda').iter_transcripts():
        conversation_id = 'sw' + str(trans.conversation_no)
        if conversation_id not in conversation_list:
            continue

        for utt in trans.utterances:
            words = ' '.join(utt.text_words(filter_disfluency=True))
            words = ''.join(filter(lambda x: x in string.printable, words))
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

            # irregular annotations
            for s in irregular_strings:
                words = re.sub(s, '', words)
            words = re.sub('<Laughter.>', '<Laughter>', words)
            words = re.sub('<Talking.>', '<Talking>', words)
            words = re.sub('<Children _talking>', '<Children_talking>', words)
            words = re.sub('<baby crying>', '<baby_crying>', words)
            words = re.sub('<child squealing>', '<child_squealing>', words)
            words = re.sub('<child-talking>', '<child_talking>', words)
            words = re.sub('<clicks on telephone>', '<clicks_on_telephone>', words)
            words = re.sub('<hiss or static>', '<hiss_or_static>', words)
            words = re.sub('<more laughing>', '<more_laughing>', words)

            if do_lowercase is True:
                words = words.lower()

            if do_pretokenization is True:
                words = re.sub(r'([.,?!":])', ' \\1 ', words)

            words = words.split()
            if len(words) == 0:
                words = ['<non_verbal>'] + words
            if len(words) == 1 and words[0] in '.,?!":':
                words = ['<non_verbal>'] + words
            words = ' '.join(words)

            if strip_punctuation is True:
                words = words.translate(str.maketrans('', '', '.,?!":'))

            user_defined_symbols.update(re.findall("<[^>]*>", words))

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
                        corpus[conversation_id]['text'][i] += ' <CONNECTOR> '+words
                        concatenated = True
                        break
                assert concatenated is True, 'please comment out the line #195 of swda/swda.py.'

    return corpus, tag_set, speaker_set, user_defined_symbols


irregular_strings = [
    "< another example like I guess it's alzheimer's >",
    '< do you think this could possibly be prohibit\? nothing elseseems to make any sense\?\? >',
    '< like as verb of saying! >',
    '< messy kind of inaccurate bracketing >',
    '< need to separate every th >',
    '< the "they" in this sentence should probably be a "the">',
    "< this is the second time the person has said 'which that' >",
    "< this looks like a typo for 'see,' >",
    '< typo incase=in case >',
    '< typo: that for than >',
    "< typo: your for you're >",
    '< typo\? I pact =impact >',
    "< what to do about these I guess's >",
    '<typo watch=watched >'
]