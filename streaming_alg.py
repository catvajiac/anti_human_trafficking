#!/usr/bin/env python3
# Author:   Catalina Vajiac
# Purpose:  Use LSH idea to cluster ad data
# Usage:    ./streaming_alg.py [filename]

import math
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os, sys
import pandas
import pickle
import random
import time

from collections import Counter, defaultdict
from datetime import datetime
from itertools import groupby
from nltk.tokenize import word_tokenize
from sortedcontainers import SortedList


DESCRIPTION = 'u_Description'
#DESCRIPTION = 'body'
TIMESTAMP = 'PostingDate'
AD_ID = 'ad_id'


class hash_family():
    def __init__(self, num_hash_functions=64, buckets_in_hash=1000):
        self.num_hash_functions = num_hash_functions
        self.hash_cutoff = num_hash_functions / 2
        self.buckets_in_hash = buckets_in_hash

        random_generator = lambda: random.randrange(buckets_in_hash)
        self.hash_functions = [defaultdict(random_generator) for _ in range(num_hash_functions)]
        self.hash_tables = [defaultdict(list) for _ in range(num_hash_functions)]


    def get_hashes(self):
        ''' return pairs of hash functions and hash tables '''
        for h, table in zip(self.hash_functions, self.hash_tables):
            yield h, table


    def add_to_hash_tables(self, to_hash, to_add):
        # TODO: SortedList? queue? no popping front, only pop booty
        for h, table in zip(self.hash_functions, self.hash_tables):
            if to_add not in table[h[to_hash]]:
                table[h[to_hash]].append(to_add)
            if len(table[h[to_hash]]) >= 1000:
                table[h[to_hash]].pop(0)


    def pretty_print(self):
        for index, table in enumerate(self.hash_tables):
            print('Table', index)
            for bucket, elements in table.items():
                print('  ', bucket, ':', ' '.join(map(str, elements)))


class data():
    def __init__(self, filename, num_phrases=5, ngrams = [3, 4, 5]):
        self.filename = os.path.basename(filename).split('.')[0]
        self.num_phrases = num_phrases
        self.ngrams = ngrams
        self.time = 0

        self.data = pandas.read_csv(filename)
        self.data.sort_values(by=['PostingDate']) # since ads not in order as they should be
        self.num_ads = len(self.data.index)
        self.cluster_graph = nx.DiGraph()
        self.hashes = hash_family()


    def preprocess_ad(self, ad):
        #return word_tokenize(ad)
        try:
            return word_tokenize(''.join([c.lower() for c in ad if c.isalnum() or c == ' ']))
        except:
            return word_tokenize('')

    def get_tokens(self, ad_text):
        ad_text = self.preprocess_ad(ad_text)
        for ngram in self.ngrams:
            for token in list(zip(*[ad_text[i:] for i in range(ngram)])):
                yield token

    def find_idf(self):
        save_path = 'pkl_files/{}_idf.pkl'.format(self.filename)
        if os.path.exists(save_path):
            self.idf = pickle.load(open(save_path, 'rb'))
            return

        print('Finding idf...')
        self.idf = Counter()
        for _, row in self.data.iterrows():
            for phrase in self.get_tokens(row[DESCRIPTION]):
                self.idf[phrase] += 1

        for phrase, doc_freq in self.idf.items():
            self.idf[phrase] = math.log10(self.num_ads/self.idf[phrase])

        pickle.dump(self.idf, open(save_path, 'wb'))


    def tfidf(self, word, document):
        tokens = list(zip(*[document[i:] for i in range(len(word))]))
        return tokens.count(word) / len(tokens) * self.idf[word]


    def calc_tfidf(self, ad_text, return_all=False):
        scores = []
        for ngram in self.ngrams:
            tokens = list(zip(*[ad_text[i:] for i in range(ngram)]))
            for index, phrase in enumerate(tokens):
                score = self.tfidf(tuple(phrase), ad_text)
                scores.append((score, index, phrase))

        scores.sort()
        if return_all:
            return scores

        # filter scores so that they take non-overlapping ngrams
        filtered_scores = []
        used_indices = set()
        deleted_phrases = []
        while len(filtered_scores) < self.num_phrases:
            if not len(scores) and not len(deleted_phrases):
                break

            if not len(scores):
                filtered_scores.append(deleted_phrases.pop(0))
                continue

            score, index, phrase = scores.pop()
            if any([i in used_indices for i in range(index, index+len(phrase))]):
                deleted_phrases.append((score, index, phrase))
                continue

            filtered_scores.append((score, index, phrase))
            used_indices.update(range(index, index + len(phrase)))

        return filtered_scores


    def find_related_clusters(self, phrases, ad_id):
        ''' return dict of related clusters, cluster type, and cluster id
            hash phrase for all k hash functions, find which clusters are related '''

        t = time.time()
        related_clusters = defaultdict(lambda: Counter())

        for phrase in phrases:
            for h, table in self.hashes.get_hashes():
                for cluster in table[h[phrase]]:
                    related_clusters[cluster][phrase] += 1

        for cluster, phrase_count in related_clusters.copy().items():
            if all([count == self.hashes.num_hash_functions for _, count in phrase_count.items()]):
                return {cluster: phrase_count}, 'dense', cluster

            if all([count <= self.hashes.num_hash_functions/2 for _, count in
                phrase_count.items()]):
                related_clusters.pop(cluster)

        self.time += (time.time() - t)
        return related_clusters, 'chain', self.cluster_graph.number_of_nodes()


    def add_new_cluster(self, related_clusters, cluster_id, ad_id):
        self.cluster_graph.add_node(cluster_id, type='dense', contains=list([ad_id]))
        self.cluster_graph.add_edges_from([(s, cluster_id) for s in related_clusters])


    def process_ad(self, row):
        ad_text = self.preprocess_ad(row[DESCRIPTION])
        ad_id = row[AD_ID]

        top_tfidf = [phrase for _, _, phrase in self.calc_tfidf(ad_text)]
        related_clusters, cluster_type, cluster_id = self.find_related_clusters(top_tfidf, ad_id)

        if cluster_type == 'chain':
            self.add_new_cluster(related_clusters, cluster_id, ad_id)
        else:
            self.cluster_graph.nodes[cluster_id]['contains'].append(ad_id)

        for phrase in top_tfidf:
            self.hashes.add_to_hash_tables(phrase, cluster_id)


    def write_cluster_graph(self):
        with open('pkl_files/{}_ad_graph.pkl'.format(self.filename), 'wb') as f:
            pickle.dump(self.cluster_graph, f)


    def clustering(self):
        t = time.time()
        self.find_idf()
        print('Finished idf in time:', time.time() - t)
        print('Starting clustering...')

        # assume in order of timestamp (streaming case)
        for index, row in self.data.iterrows():
            if index and not index % 1000:
                print(index, '/', self.num_ads, 'time', time.time() - t)
                print('\t', self.time)
                self.write_cluster_graph()

            self.process_ad(row)

        self.write_cluster_graph()
        print('Finished clustering!', time.time() - t)
        self.visualize_buckets()

    def visualize_buckets(self):
        print('Plotting hash tables...')
        save_path = './plots/streaming_alg/' + self.filename
        print(save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        length = self.cluster_graph.number_of_nodes()
        for index in range(self.hashes.num_hash_functions):
            hash_table = self.hashes.hash_tables[index]
            m = np.zeros((len(hash_table), length))
            for i, (_, cluster_ids) in enumerate(hash_table.items()):
                m[i, cluster_ids] = 1

            plt.imshow(m, interpolation='nearest', aspect='auto')
            plt.tight_layout()
            plt.title('Hash table: {}'.format(index))
            plt.xlabel('cluster ids')
            plt.ylabel('buckets, sorted by first access time')
            plt.savefig('{}/{}_hash_visual.png'.format( save_path, index))
            plt.clf()


def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    filename = sys.argv[1]
    canadian_data = data(filename)
    canadian_data.clustering()
