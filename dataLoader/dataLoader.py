import argparse
import hashlib
import pickle, tarfile
import random
import re
import os
import numpy as np

class Document():
    def __init__(self, content, summary):
        self.content = content
        self.summary = summary

class Dataset():
    def __init__(self, data_list):
        self._data = data_list

    def __len__(self):
        return len(self._data)

    def __call__(self, batch_size, shuffle=True):
        max_len = len(self)
        if shuffle:
            random.shuffle(self._data)
        batchs = [self._data[index:index + batch_size] for index in range(0, max_len, batch_size)]
        return batchs

    def __getitem__(self, index):
        return self._data[index]

class Vocab():
    def __init__(self):

        #  PAD_ID = 0
        #  UNK_ID = 1
        #  PAD_TOKEN = '<pad>'
        #  UNK_TOKEN = '<unk>'
        self.word_list = ['<pad>', '<unk>', '<s>', '<\s>']
        self.word2idx = {}
        self.idx2word = {}
        self.count = 0
        self.embedding = None

    def __getitem__(self, key):
        if self.word2idx.has_key(key):
            return self.word2idx[key]
        else:
            return self.word2idx['<unk>']


    def add_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            for line in f:
                self.word_list.append(line.split()[0]) # only want the word, not the count
            print ('read %d words from vocab file' % len(self.word_list))

        for w in self.word_list:
            self.word2idx[w] = self.count
            self.idx2word[self.count] = w
            self.count += 1

    def add_embedding(self, embed_file, embed_size):
        print('Loading embeddings ')
        with open(embed_file, 'r') as f:
            word_set = set(self.word_list)
            embed_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            count = 0
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in word_set:
                    count += 1
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    embed_matrix[self.word2idx[word]] = embedding

                if count % 1000 == 0:
                    print('processed %d data' % count)

            self.embeddings = embed_matrix
            print('%d words out of %d has embeddings in the embed_file' % (count, len(self.word_list)))



def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""

    dm_single_close_quote = u'\u2019'  # unicode
    dm_double_close_quote = u'\u201d'
    END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
                  ")"]  # acceptable ways to end a sentence

    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line

    return line + " ."

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

def get_url_hashes(url_list):
    return [hashhex(url.encode('utf-8')) for url in url_list]


def build_dataset(args):
    
    # get article and abstract from story_file
    def get_art_abs(story_file):
        lines = read_text_file(story_file)

        # Lowercase everything
        lines = [line.lower() for line in lines]

        # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
        lines = [fix_missing_period(line) for line in lines]

        # Separate out article and abstract sentences
        article_lines = []
        highlights = []
        next_is_highlight = False
        for idx, line in enumerate(lines):
            if line == "":
                continue  # empty line
            elif line.startswith("@highlight"):
                next_is_highlight = True
            elif next_is_highlight:
                highlights.append(line)
            else:
                article_lines.append(line)

        # Make article into a single string
        article = ' '.join(article_lines)

        # Make abstract into a signle string, putting <s> and </s> tags around the sentences
        abstract = ' '.join(["%s %s %s" % ('<s>', sent, '</s>') for sent in highlights])

        return article.split(' '), abstract.split(' ') # split sentence to words 


    def write_to_pickle(url_file, source_dir, target_dir, chunk_size = 10000):


        url_list = read_text_file(url_file)
        url_hashes = get_url_hashes(url_list) # the name of every story is mapped by hashhex
        url = zip(url_list, url_hashes)

        story_fnames = [source_dir + 'cnn_stories_tokenized/' + s +'.story'
                if u.find('cnn.com') >= 0 else (source_dir + 'dm_stories_tokenized/' + s + '.story')
                for u, s in url]

        new_lines = []
        for i, filename in enumerate(story_fnames):
            if i % chunk_size == 0 and i > 0:
                pickle.dump(Dataset(new_lines), open(target_dir % (i / chunk_size), "wb"))
                print ('%d samples have been processed.' % (i))
                new_lines = []

            try:
                article, abstract = get_art_abs(filename)
            except:
                print(filename)
                continue
            new_lines.append(Document(article, abstract))

        if new_lines != []:
            pickle.dump(Dataset(new_lines), open(target_dir % (i / chunk_size + 1), "wb"))


    print (args)
    print('start building Dataset')

    train_urls = ''.join((args.url_lists, 'train.txt'))
    test_urls = ''.join((args.url_lists, 'test.txt'))
    val_urls = ''.join((args.url_lists, 'val.txt'))

    target_dir = args.target_dir + 'chunked/'
    try:
        os.makedirs(target_dir)
    except OSError:
        if not os.path.isdir(target_dir):
            print ('can not create target_dir! ')

    write_to_pickle(train_urls,
            args.source_dir,
            ''.join((target_dir, 'train_%03d.pickle')),
            chunk_size = 10000)

    write_to_pickle(test_urls, 
            args.source_dir, 
            ''.join((target_dir, 'test_%03d.pickle')),
            chunk_size = 10000)

    write_to_pickle(val_urls, 
            args.source_dir, 
            ''.join((target_dir, 'val_%03d.pickle')),
            chunk_size = 10000)

def build_vocab(args):
    print(args)
    print('start building vocab')

    vocab = Vocab()
    vocab.add_vocab(args.vocab_file)
    vocab.add_embedding(args.embed_file, args.embed_size)

    pickle.dump(vocab, open(args.finished_vocab, 'wb'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-source_dir', type=str, default='../data/cnn_dailymail_data/')
    parser.add_argument('-target_dir', type=str, default='../data/cnn_dailymail_data/finished_dm_data/')
    parser.add_argument('-url_lists', type=str, default='../data/cnn_dailymail_data/url_lists/dm_urls/')

    # if you want to build vocab, below are some parameters that you should considern
    parser.add_argument('-build_vocab', action='store_true', default=False)
    parser.add_argument('-vocab_file', type=str, default='../data/cnn_dailymail_data/vocab')
    parser.add_argument('-embed_file', type=str, default='../data/cnn_dailymail_data/glove.6B.100d.txt')
    parser.add_argument('-embed_size', type=int, default=100)
    parser.add_argument('-finished_vocab', type=str, default='../data/cnn_dailymail_data/finished_dm_data/vocab_file.pickle')

    args = parser.parse_args()

    if args.build_vocab:
        build_vocab(args)
    else:
        build_dataset(args)
