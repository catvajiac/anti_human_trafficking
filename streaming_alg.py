#!/usr/bin/env python3

import math
import networkx as nx
import os, sys
import pandas
import pickle
import random
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from datetime import datetime

class hash_family():
    def __init__(self, num_hash_functions=32, buckets_in_hash=1000):
        self.num_hash_functions = num_hash_functions
        self.buckets_in_hash = buckets_in_hash

        random_generator = lambda: random.randrange(buckets_in_hash)
        self.hash_functions = [defaultdict(random_generator) for _ in range(num_hash_functions)]
        self.hash_tables = [defaultdict(list) for _ in range(num_hash_functions)]

    def add_to_hash_tables_and_graph(self, phrase, ad_id, data):
        related_ads = Counter()
        # hash phrase for all k hash functions
        for h, table in zip(self.hash_functions, self.hash_tables):
            for ad in table[h[phrase]]:
                related_ads[ad] += 1
            if len(table[h[phrase]]) >= 1000:
                table[h[phrase]].pop(0)
            table[h[phrase]].append(ad_id)

        # draw edges
        edges = [(s, ad_id, count) for s, count in related_ads.items() if count >= self.num_hash_functions / 2]
        data.ad_graph.add_weighted_edges_from(edges)


    def pretty_print(self):
        for index, table in enumerate(self.hash_tables):
            print('Table', index)
            for bucket, elements in table.items():
                print('  ', bucket, ':', ' '.join(map(str, elements)))


class data():
    def __init__(self, filename, num_phrases=2):
        self.filename = filename
        self.num_phrases = num_phrases
        self.data = pandas.read_csv(self.filename)

    def find_idf(self):
        print('Finding idf...')
        idf = Counter()
        for _, row in self.data.iterrows():
            for word in set(row['description'].split()):
                idf[word] += 1

        N = len(self.data.index)
        for word, doc_freq in idf.items():
            idf[word] = math.log10(idf[word]/(N+1))

        self.idf = idf

    def process_data(self):
        print('Processing ads...')
        def tfidf(word, document):
            return document.count(word) / len(document) * self.idf[word]

        self.find_idf()
        hashes = hash_family()
        self.ad_graph = nx.DiGraph()

        N = len(self.data.index)
        for index, row in self.data.iterrows():
            # write graph every 1000 ads
            if index % 1000 == 0 and index != 0:
                print(index, '/', N)
                with open('ad_graph2.pkl', 'wb') as f:
                    pickle.dump(self.ad_graph, f)

            ad_text = row['description'].split()

            tfidf_scores = [(tfidf(word, ad_text), word) for word in ad_text]
            tfidf_scores.sort(reverse=True)

            for score, word in tfidf_scores[:self.num_phrases]:
                hashes.add_to_hash_tables_and_graph(word, index, self)



def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    mode = sys.argv[1]

    if not os.path.isdir('./plots'):
        os.mkdir('./plots')

    filename = sys.argv[1]
    canadian_data = data(filename)

    canadian_data.process_data()
