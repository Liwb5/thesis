import os
import sys
import hashlib
import collections
from tqdm import tqdm
import json

all_train_urls = "../data/cnn-dailymail/url_lists/all_train.txt"
all_val_urls = "../data/cnn-dailymail/url_lists/all_val.txt"
all_test_urls = "../data/cnn-dailymail/url_lists/all_test.txt"

cnn_tokenized_stories_dir = "../data/cnn-dailymail/cnn_stories_tokenized"
dm_tokenized_stories_dir = "../data/cnn-dailymail/dm_stories_tokenized"
finished_files_dir = "../data/cnn-dailymail/finished_files2/"

VOCAB_SIZE = 200000
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url.encode('utf-8')) for url in url_list]

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

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
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    #  is_article = True
    #  if len(article_lines) == 0 or len(highlights) == 0:
    #      is_article = False
    # Make article into a single string
    #  article = '\n'.join(article_lines)
    #  abstract = '\n'.join(highlights)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    #  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article_lines, highlights #, is_article

def judge(article, abstract, filename):
    """
    article: list of string, every element is a sentence
    abstract: list of string
    filename: string, filename of article
    """
    is_article = True

    if len(article) == 0 or len(abstract) == 0: # some article is empty
        print('%s file is empty. '%(filename))
        is_article = False
        return is_article

    a = ' '.join(article)
    a = a.split(' ')
    b = ' '.join(abstract)
    b = b.split(' ')
    if len(a) < 10 or len(b) < 10:  # the content of an article is too little
        print('The content of %s file is too little. '%(filename))
        is_article = False
        return is_article

    return is_article

def write_to_json(url_file, out_file, makevocab=False):
    url_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)
    #  print('number of stories is %d'%num_stories)
    #  print(story_fnames[0])

    if makevocab:
        vocab_counter = collections.Counter()

    res = []
    for idx, s in enumerate(story_fnames):
        if idx % 1000 == 0:
            print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories))) 
        #  print(os.path.join(cnn_tokenized_stories_dir, s))
        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
            story_file = os.path.join(cnn_tokenized_stories_dir, s)
        elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
            story_file = os.path.join(dm_tokenized_stories_dir, s)
        else:
            print("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir)) 

        article, abstract = get_art_abs(story_file)
        is_article = judge(article, abstract, s)
        if is_article == False:
            #  print('story file %s is empty'%(s))
            continue
        else:
            article = '\n'.join(article)
            abstract = '\n'.join(abstract)

        res.append({'doc':article, 'summaries':abstract, 'labels': '1\n0'})

        # Write the vocab to file, if applicable
        if makevocab:
            art_sents = article.split('\n')
            art_tokens = []
            for sent in art_sents:
                art_tokens += [word for word in sent.split(' ')]

            abs_sents = abstract.split('\n')
            abs_tokens = []
            for sent in abs_sents:
                abs_tokens += [word for word in sent.split(' ')]

            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens] # strip
            tokens = [t for t in tokens if t!=""] # remove empty
            vocab_counter.update(tokens)

    print('%d articles has been processed in %s'%(len(res), out_file))
    with open(out_file, 'w') as writer:
        for item in res:
            obj = json.dumps(item)
            writer.write(obj)
            writer.write('\n')

    # write vocab to file
    if makevocab:
        print("Writing vocab file...") 
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file") 

if __name__=='__main__':

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
    write_to_json(all_train_urls, os.path.join(finished_files_dir, 'train.json'), makevocab=True)
    write_to_json(all_val_urls, os.path.join(finished_files_dir, 'val.json'))
    write_to_json(all_test_urls, os.path.join(finished_files_dir, 'test.json'))
