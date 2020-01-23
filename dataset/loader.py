import os
import re
import string
from numpy import loadtxt
from collections import defaultdict
from dataset.swda.swda import CorpusReader


def load_mrda_corpus(conversation_list,
                     strip_punctuation,
                     tokenize_punctuation,
                     tag_map):
    """ Load MRDA corpus based on https://github.com/NathanDuran/MRDA-Corpus
    :param conversation_list:
    :param strip_punctuation:
    :param tokenize_punctuation:
    :param tag_map: basic, general, full.
    :return:
    """
    corpus = defaultdict(lambda: defaultdict(list))
    tag_set = set()
    speaker_set = set()
    symbol_set = set()

    for conversation_id in conversation_list:
        for path in ['dataset/MRDA-Corpus/mrda_data/'+folder+'/'+conversation_id+'.txt'
                     for folder in ['train', 'dev', 'eval', 'test']]:
            if os.path.isfile(path):
                trans = loadtxt(path, delimiter='|', dtype=str)
                col_0, col_1, col_2, col_3, col_4 = (list(trans[:, i]) for i in range(5))

                corpus[conversation_id]['speaker'] = col_0
                speaker_set.update(set(col_0))

                sentence_list = col_1
                if strip_punctuation is True:
                    sentence_list = [
                        sentence.translate(str.maketrans('', '', '.,?!":'))
                        for sentence in sentence_list
                    ]
                if tokenize_punctuation is True:
                    sentence_list = [
                        ' '.join(re.sub(r'([.,?!":])', ' \\1 ', sentence).split(' '))
                        for sentence in sentence_list
                    ]
                corpus[conversation_id]['sentence'] = sentence_list

                if tag_map == 'basic':
                    tag_list = col_2
                if tag_map == 'general':
                    tag_list = col_3
                if tag_map == 'full':
                    tag_list = col_4
                corpus[conversation_id]['tag'] = tag_list
                tag_set.update(set(tag_list))

                break

    return corpus, tag_set, speaker_set, symbol_set


def load_swda_corpus(conversation_list,
                     strip_punctuation,
                     tokenize_punctuation,
                     concatenate_interruption,
                     do_lowercase=True):
    """ Load SwDA corpus based on https://github.com/cgpotts/swda
    :param conversation_list:
    :param strip_punctuation:
    :param tokenize_punctuation:
    :param concatenate_interruption:
    :param do_lowercase:
    :return:
    """
    corpus = defaultdict(lambda: defaultdict(list))
    tag_set = set()
    speaker_set = set()
    symbol_set = set()
    symbol_set.add('<CONNECTOR>')

    for trans in CorpusReader('dataset/swda/swda').iter_transcripts():
        conversation_id = 'sw' + str(trans.conversation_no)
        if conversation_id not in conversation_list:
            continue

        for utt in trans.utterances:
            sentence = ' '.join(utt.text_words(filter_disfluency=True))
            sentence = re.sub(r'<<[^>]*>>', '', sentence)
            sentence = re.sub(r'\*.*$', '', sentence)
            sentence = re.sub(r'[#)(]', '', sentence)
            sentence = re.sub(r'--', '', sentence)
            sentence = re.sub(r' -$', '', sentence)
            while True:
                output = re.sub(r'([A-Z]) ([A-Z])\b', '\\1\\2', sentence)
                if output == sentence:
                    break
                sentence = output

            # irregular annotations
            for irr_str in swda_irregular_annotation_strings:
                sentence = re.sub(irr_str, '', sentence)
            sentence = re.sub('<Laughter.>', '<Laughter>', sentence)
            sentence = re.sub('<Talking.>', '<Talking>', sentence)
            sentence = re.sub('<Children _talking>', '<Children_talking>', sentence)
            sentence = re.sub('<baby crying>', '<baby_crying>', sentence)
            sentence = re.sub('<child squealing>', '<child_squealing>', sentence)
            sentence = re.sub('<child-talking>', '<child_talking>', sentence)
            sentence = re.sub('<clicks on telephone>', '<clicks_on_telephone>', sentence)
            sentence = re.sub('<hiss or static>', '<hiss_or_static>', sentence)
            sentence = re.sub('<more laughing>', '<more_laughing>', sentence)

            if do_lowercase is True:
                sentence = sentence.lower()

            if strip_punctuation is True:
                sentence = sentence.translate(str.maketrans('', '', '.,?!":'))

            if tokenize_punctuation is True:
                sentence = re.sub(r'([.,?!":])', ' \\1 ', sentence)

            sentence = sentence.split()
            if len(sentence) == 0:
                sentence = ['<non_verbal>']
            if len(sentence) == 1 and sentence[0] in '.,?!":':
                sentence = ['<non_verbal>'] + sentence
            sentence = ' '.join(sentence)

            symbol_set.update(re.findall("<[^>]*>", sentence))

            utt_tag = utt.damsl_act_tag()
            if (utt_tag != '+') or (utt_tag == '+' and concatenate_interruption is False):
                corpus[conversation_id]['sentence'].append(sentence)
                corpus[conversation_id]['tag'].append(utt_tag)
                corpus[conversation_id]['speaker'].append(utt.caller)
                tag_set.add(utt_tag)
                speaker_set.add(utt.caller)
            # resolve issue related to tag '+' (check section 2 of the Coders' Manual)
            else:
                concatenated = False
                for i in reversed(range(len(corpus[conversation_id]['tag']))):
                    if corpus[conversation_id]['speaker'][i] == utt.caller:
                        corpus[conversation_id]['sentence'][i] += ' <CONNECTOR> '+sentence
                        concatenated = True
                        break
                assert concatenated is True, 'please comment out the line #195 of dataset/swda/swda.py.'

    return corpus, tag_set, speaker_set, symbol_set


swda_irregular_annotation_strings = [
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

tag_name_dict = {
'swda': {
 '%': 'Abandoned or Turn-Exit, Uninterpretable',
 '^2': 'Collaborative Completion',
 '^g': 'Tag-Question',
 '^h': 'Hold before answer/agreement',
 '^q': 'Quotation',
 'aa': 'Agree/Accept',
 'aap_am': 'Maybe/Accept-part',
 'ad': 'Action-directive',
 'ar': 'Reject',
 'arp_nd': 'Dispreferred answers',
 'b': 'Acknowledge (Backchannel)',
 'b^m': 'Repeat-phrase',
 'ba': 'Appreciation',
 'bd': 'Downplayer',
 'bf': 'Summarize/reformulate',
 'bh': 'Backchannel in question form',
 'bk': 'Response Acknowledgement',
 'br': 'Signal-non-understanding',
 'fa': 'Apology',
 'fc': 'Conventional-closing',
 'fo_o_fw_"_by_bc': 'Other',
 'fp': 'Conventional-opening',
 'ft': 'Thanking',
 'h': 'Hedge',
 'na': 'Affirmative non-yes answers',
 'ng': 'Negative non-no answers',
 'nn': 'No answers',
 'no': 'Other answers',
 'ny': 'Yes answers',
 'oo_co_cc': 'Offers, Options, Commits',
 'qh': 'Rhetorical-Questions',
 'qo': 'Open-Question',
 'qrr': 'Or-Clause',
 'qw': 'Wh-Question',
 'qw^d': 'Declarative Wh-Question',
 'qy': 'Yes-No-Question',
 'qy^d': 'Declarative Yes-No-Question',
 'sd': 'Statement-non-opinion',
 'sv': 'Statement-opinion',
 't1': 'Self-talk',
 't3': '3rd-party-talk',
 'x': 'Non-verbal'
},
'mrda': {
  'B': 'BackChannel',
  'D': 'Disruption',
  'F': 'FloorGrabber',
  'Q': 'Question',
  'S': 'Statement'
}
}
