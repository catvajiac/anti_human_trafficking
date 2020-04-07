#!/usr/bin/env python3
# Author:   Catalina Vajiac
# Purpose:  Use LSH idea to cluster ad data
# Usage:    ./streaming_alg.py [filename]

import math
import networkx as nx
import os, sys
import pandas
import pickle
import random

from collections import Counter, defaultdict
from datetime import datetime
from nltk.tokenize import word_tokenize


DESCRIPTION = 'u_Description'
TIMESTAMP = 'PostingDate'
AD_ID = 'ad_id'


class hash_family():
    def __init__(self, num_hash_functions=256, buckets_in_hash=5000):
        self.num_hash_functions = num_hash_functions
        self.hash_cutoff = num_hash_functions / 2
        self.buckets_in_hash = buckets_in_hash

        random_generator = lambda: random.randrange(buckets_in_hash)
        self.hash_functions = [defaultdict(random_generator) for _ in range(num_hash_functions)]
        self.hash_tables = [defaultdict(list) for _ in range(num_hash_functions)]


    def add_to_hash_tables(self, to_hash, to_add):
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
        self.filename = filename
        self.num_phrases = num_phrases
        self.ngrams = ngrams

        self.data = pandas.read_csv(self.filename)
        self.data.sort_values(by=['PostingDate']) # since ads not in order as they should be
        self.num_ads = len(self.data.index)
        self.cluster_graph = nx.DiGraph()
        self.hashes = hash_family()


    def preprocess_ad(self, ad):
        return word_tokenize(''.join([c for c in ad if c.isalnum() or c==' ']))


    def find_idf(self):
        print('Finding idf...')
        self.idf = Counter()
        for _, row in self.data.iterrows():
            ad_text = self.preprocess_ad(row[DESCRIPTION])
            for ngram in self.ngrams:
                tokens = list(zip(*[ad_text[i:] for i in range(ngram)]))

                for phrase in set(tokens):
                    self.idf[phrase] += 1

        for phrase, doc_freq in self.idf.items():
            self.idf[phrase] = math.log10(self.num_ads/self.idf[phrase])


    def calc_tfidf(self, ad_text):
        def tfidf(word, document):
            tokens = list(zip(*[document[i:] for i in range(len(word))]))
            return tokens.count(word) / len(tokens) * self.idf[word], word

        scores = []
        for ngram in self.ngrams:
            tokens = list(zip(*[ad_text[i:] for i in range(ngram)]))
            for phrase in tokens:
                scores.append(tfidf(tuple(phrase), ad_text))

        scores.sort(reverse=True)
        return scores[:self.num_phrases]


    def find_related_clusters(self, phrases, ad_id):
        related_clusters = [Counter() for _ in phrases]
        # hash phrase for all k hash functions, find which clusters are related
        cluster_type = 'chain'
        cluster_id = self.cluster_graph.number_of_nodes()
        for i, phrase in enumerate(phrases):
            for h, table in zip(self.hashes.hash_functions, self.hashes.hash_tables):
                for cluster in table[h[phrase]]:
                    related_clusters[i][cluster] += 1
                    # if all phrases map to same cluster, then text is identical whp
                    if all([rel[cluster] == self.hashes.num_hash_functions for rel in related_clusters]):
                        cluster_type = 'dense'
                        cluster_id = cluster

        return related_clusters, cluster_type, cluster_id


    def add_new_cluster(self, related_clusters, cluster_id, ad_id):
        self.cluster_graph.add_node(cluster_id, type='dense', contains=list([ad_id]))
        edges = []
        for related_cluster in related_clusters:
            edges += [(s, cluster_id) for s, num in related_cluster.items() if num >= self.hashes.hash_cutoff]
        self.cluster_graph.add_edges_from(edges)


    def process_ad(self, row):
        ad_text = self.preprocess_ad(row[DESCRIPTION])
        ad_id = row[AD_ID]

        top_tfidf = [phrase for _, phrase in self.calc_tfidf(ad_text)]
        related_clusters, cluster_type, cluster_id = self.find_related_clusters(top_tfidf, ad_id)
        if cluster_type == 'chain':
            self.add_new_cluster(related_clusters, cluster_id, ad_id)
        else:
            self.cluster_graph.nodes[cluster_id]['contains'].append(ad_id)

        for phrase in top_tfidf:
            self.hashes.add_to_hash_tables(phrase, cluster_id)


    def write_cluster_graph(self):
        with open('{}_ad_graph.pkl'.format(filename), 'wb') as f:
            pickle.dump(self.cluster_graph, f)


    def clustering(self):
        self.find_idf()
        print('Starting clustering...')

        # assume in order of timestamp (streaming case)
        for index, row in self.data.iterrows():
            if index and not index % 1000:
                print(index, '/', self.num_ads)
                self.write_cluster_graph()

            self.process_ad(row)

        self.write_cluster_graph()


def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    filename = sys.argv[1]
    canadian_data = data(filename)
    canadian_data.clustering()
