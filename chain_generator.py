#!/usr/bin/env python3
# Author: Catalina
# Purpose: generate chain csv data for experiments for AHT
# Usage: ./chain_generator.py


import copy
import datetime
import math
import os, sys
import pandas
import random

from collections import defaultdict
from nltk.corpus import words

WORDS = words.words()


def gen_data(text_length=50, num_ads=100, num_parallel_chains=1, percent_change=0.1):
    ''' generates data with above params, all chains start from one source ad '''
    original_text = [random.choice(WORDS) for _ in range(text_length)]
    corpus = defaultdict(list)
    for _ in range(num_parallel_chains):
        corpus = gen_one_chain(copy.deepcopy(original_text), corpus, text_length, num_ads, percent_change)

    save_path = './data/random_chain-{}.csv'.format(num_ads)
    pandas.DataFrame.from_dict(corpus).to_csv(save_path, encoding='utf-8', index_label='ad_id')


def gen_one_chain(text, corpus, text_length, num_ads, percent_change):
    date = datetime.datetime(day=1, month=1, year=2010, hour=00, minute=00)

    for _ in range(num_ads):
        corpus['PostingDate'].append(date)
        corpus['Url'] = ''
        corpus['l_Location'] = 'USA'
        corpus['u_Age'] = 20
        corpus['u_Description'].append(' '.join(text))
        corpus['u_Title'] = 'hi'
        corpus['u_Location'] = 'USA'
        corpus['e_PhoneNumbers'] = 123
        corpus['u_PhoneNumbers'] = 234
        corpus['e_Emails'] = 'fake@abc'

        date += datetime.timedelta(days=1)
        num_substitutions = random.randrange(math.ceil(percent_change*len(text)))
        for _ in range(num_substitutions):
            index = random.choice(range(len(text)))
            text[index] = random.choice(WORDS)

    return corpus


def usage(code):
    print('Usage: {}')
    exit(code)


if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')

    gen_data()
