from io import open
import csv
import unicodedata
import regex  
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from model_translation_architecture import EncoderRNN,AttnDecoderRNN
from config import * 


HIDDEN_DIM = hidden_size
BATCH_SIZE = batch_size

ENCODER_WEIGHT_PATH = ENCODER_WEIGHT_PATH_eng_ara
DECODER_WEIGHT_PATH = DECODER_WEIGHT_PATH_eng_ara

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EOS_TOKEN = 1  # End of sentence

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_arabic(text):

    text = unicode_to_ascii(text.lower().strip())
    # Normalize Arabic characters
    text = regex.sub(r'[\p{Mn}\p{Sk}]+', '', unicodedata.normalize('NFKD', text))

    # Remove non-letter, non-space characters
    text = regex.sub(r'[^\p{L}\s]', '', text)

    # Normalize whitespace
    text = regex.sub(r'\s+', ' ', text)

    return text.strip()

def read_languages(lang1, lang2, reverse=False):
    print("preparing vocab ...")

    # Open the CSV file
    with open(translation_data, newline='', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)
        
        # Read the rows of the CSV file
        lines = [row for row in reader]

    # Split every line into pairs and normalize
    pairs = [[normalize_arabic(s) for s in l] for l in lines]

    # Reverse pairs, make Language instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang = Language(lang1)
        output_lang = Language(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = MAX_LENGTH

ARA_PREFIXES = ( 'هذه','انا', 'لا')

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(ARA_PREFIXES) # turning off filtering 

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_languages(lang1, lang2, reverse)
    pairs = filter_pairs(pairs)
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('english', 'arabic', False)

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(1, -1)

def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepare_data('eng', 'ara', False)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexes_from_sentence(input_lang, inp)
        tgt_ids = indexes_from_sentence(output_lang, tgt)
        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(DEVICE),
                                torch.LongTensor(target_ids).to(DEVICE))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


input_lang1, output_lang1, train_dataloader = get_dataloader(BATCH_SIZE)

encoder1 = EncoderRNN(input_lang.n_words, HIDDEN_DIM).to(DEVICE)
decoder1 = AttnDecoderRNN(HIDDEN_DIM, output_lang.n_words).to(DEVICE)

encoder1.load_state_dict(torch.load(ENCODER_WEIGHT_PATH))
decoder1.load_state_dict(torch.load(DECODER_WEIGHT_PATH))

def evaluate_specific_sentence_eng_ara(encoder1, decoder1, sentence, input_lang, output_lang):
    output_words, _ = evaluate(encoder1, decoder1, sentence, input_lang, output_lang)
    output_sentence = ' '.join(output_words)
    return output_sentence
