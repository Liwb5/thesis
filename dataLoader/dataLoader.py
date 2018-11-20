import argparse


PAD_ID = 0
UNK_ID = 1
wordembed_size = 200

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


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

        return article.split(' '), abstract.split(' ')


    def write_to_pickle(url_file, source_dir, out_file, chunk_size = 10000):
        url_list = read_text_file(url_file)
        url_hashes = get_url_hashes(url_list) # the name of every story is mapped by hashhex
        url = zip(url_list, url_hashes)

        story_fnames = [source_dir + 'cnn_stories_tokenized/' + s +'.story'
                if u.find('cnn.com') >= 0 else (source_dir + 'dm_stories_tokenized/' + s + '.story')
                for u, s in url]

        new_lines = []
        for i, filename in enumerate(story_fnames):
            if i % chunk_size == 0 and i > 0:
                pickle.dump(Dataset(new_lines), open(out_file % (i / chunk_size), "wb"))
                new_lines = []

            try:
                art, abs = get_art_abs(filename)
            except:
                print filename
                continue
            new_lines.append(Document(art, abs))

        if new_lines != []:
            pickle.dump(Dataset(new_lines), open(out_file % (i / chunk_size + 1), "wb"))


    print (args)
    
    train_urls = ''.join((args.url_lists, 'train.txt'))
    test_urls = ''.join((args.url_lists, 'test.txt'))
    val_urls = ''.join((args.url_lists, 'val.txt'))


    write_to_pickle(train_urls, 
            ''.join((args.target_dir, 'chunked/train_%03d.pickle')),
            chunk_size = 10000)

    write_to_pickle(train_urls, 
            ''.join((args.target_dir, 'chunked/train_%03d.pickle')),
            chunk_size = 10000)

    write_to_pickle(train_urls, 
            ''.join((args.target_dir, 'chunked/train_%03d.pickle')),
            chunk_size = 10000)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-source_dir', type=str, default='../data/cnn_dailymail_data/')
    parser.add_argument('-target_dir', type=str, default='../data/cnn_dailymail_data/finished_cnn_dm_data/')
    parser.add_argument('-url_lists', type=str, default='../data/cnn_dailymail_data/url_lists/all_urls/')
    parser.add_argument('-worker_num', type=int, default=1)

    args = parser.parse_args()

    build_dataset(args)
