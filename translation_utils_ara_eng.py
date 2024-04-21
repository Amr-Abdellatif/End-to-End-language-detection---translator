import random
from io import open
import unicodedata
import random
import csv
import unicodedata
import regex  
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from model_translation_architecture import EncoderRNN ,AttnDecoderRNN
from config import *


HIDDEN_DIM = hidden_size
BATCH_SIZE = batch_size

encoder_weight_path_ar_en = ENCODER_WEIGHT_PATH_ara_eng
decoder_weight_path_ar_en = DECODER_WEIGHT_PATH_ara_eng

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EOS_token = 1 # end of sentence

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeArabic(text):

    text = unicodeToAscii(text.lower().strip())
    # Normalize Arabic characters
    text = regex.sub(r'[\p{Mn}\p{Sk}]+', '', unicodedata.normalize('NFKD', text))

    # Remove non-letter, non-space characters
    text = regex.sub(r'[^\p{L}\s]', '', text)

    # Normalize whitespace
    text = regex.sub(r'\s+', ' ', text)

    return text.strip()


def readLangs(lang1, lang2, reverse=True):
    print("Reading lines...")

    # Open the CSV file
    with open(translation_data, newline='', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)
        
        # Read the rows of the CSV file
        lines = [row for row in reader]

    # Split every line into pairs and normalize
    pairs = [[normalizeArabic(s) for s in l] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        # print(input_lang)
        output_lang = Lang(lang1)
        # print(output_lang)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# some filtering based on sentence length and sentence start prefix
MAX_LENGTH = MAX_LENGTH 

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ","This","That"
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes) # filtering 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=True):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('english', 'arabic', True)

    
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('ara', 'eng', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                                torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

encoder.load_state_dict(torch.load(encoder_weight_path_ar_en))
decoder.load_state_dict(torch.load(decoder_weight_path_ar_en))

def evaluateSpecificSentence_ara_eng(encoder, decoder, sentence, input_lang, output_lang):
    output_words, _ = evaluate(encoder, decoder, sentence, input_lang, output_lang)
    output_sentence = ' '.join(output_words)
    return output_sentence
    

