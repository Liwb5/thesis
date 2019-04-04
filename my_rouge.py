# commmand : python rouge.py -dir ./outputs/lead3/  > ./outputs/lead3/result.txt 2>&1 &

from pyrouge import Rouge155
import argparse

def rouge155(args):
    ROUGE_PATH = '/data1/liwb/softwares/nlp-metrics/ROUGE-1.5.5/'
    r = Rouge155(ROUGE_PATH)
    r.system_dir = args.dir + 'hyp%s/'%(args.epoch)
    r.model_dir = args.dir + 'ref%s/'%(args.epoch)
    r.system_filename_pattern = 'hyp.(\d+).txt'
    r.model_filename_pattern = 'ref.A.#ID#.txt'

    if args.b_limit:
        print('using byte limit. ')
        output = r.convert_and_evaluate(
                rouge_args='-e {}/data -a -c 95 -m -n 2 -b 275'.format(ROUGE_PATH))
                #  rouge_args='-e {}/data -a -2 -1 -c 95 -U -n 2 -w 1.2 -b 275'.format(ROUGE_PATH))
    else:
        print('not using byte limit. ')
        output = r.convert_and_evaluate()

    print(output)
    output_dict = r.output_to_dict(output)
    print(output_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rouge')
    parser.add_argument('-dir', default=None, type=str)
    parser.add_argument('-epoch', default=None, type=str)
    parser.add_argument('-b_limit', action='store_true')
    args = parser.parse_args()
    rouge155(args)

