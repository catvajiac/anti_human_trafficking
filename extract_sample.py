#!/usr/bin/env python3
# Author: Catalina Vajiac
# Purpose: extract one month's data from large data
# Usage: ./extract.py

import os, sys
import pandas
from datetime import datetime


def read_data(filename):
    return pandas.read_csv(filename)


def usage(code):
    print('Usage: {}'.format(os.path.basename(sys.argv[0])))
    exit(code)


def extract_date(data):
    ''' extract data within a certain date '''
    with open('less_data.csv', 'w') as f:
        indices = []
        count = 0
        for i, line in data.iterrows():
            date = datetime.strptime(line['PostingDate'], '%a %d %b %Y %I:%M %p')
            #Thu 14 Apr 2016 11:40 AM
            if date.year == 2016 and date.month == 12:
                count += 1
                indices.append(i)


            if count and not count % 1000:
                print(count, i)

        subset_data = data.loc[indices, :]
        subset_data.to_csv('sample.csv', encoding='utf-8', index=False)

    print('Final', count)


def extract_sample(data, ad_file):
    ''' return csv file of only ad_ids in ad_file. assume ad_file is ad_ids split by newlines'''
    with open(ad_file, 'r') as f:
        ads = [int(line) for line in f]

        subset_data = (data.loc[data['ad_id'].isin(ads)])
        subset_data.to_csv('subsample.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    data = read_data(sys.argv[1])
    #extract_date(data)
    extract_sample(data, sys.argv[2])


