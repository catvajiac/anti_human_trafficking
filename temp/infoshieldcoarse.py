#!/usr/bin/env python3
# Author:   Catalina Vajiac
# Purpose:  Use LSH idea to cluster text data
# Usage:    ./streaming_alg.py [filename]

import heapq
import math
import networkx as nx
import numpy as np
import os, sys
import pandas
import pickle
import random
import re
import scipy
import time
import string

import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from datetime import datetime
from networkx.algorithms import bipartite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score
from io import StringIO

def filter_text(text):
    if type(text) is not str: # nan
        return ''

    replace = [(r'\d+', ''), (r'[^\x00-\x7F]+', ' ')]
    replace = [(r'\d+', ''), (re.compile('<.*?>'), ' ')]
    nbsp_variants = [('&nbsp;', ''), ('nbsp;', '')]
    br_variants = [('<b', ''), ('<br', ''), ('br>', '')]
    for source, target in replace + nbsp_variants + br_variants:
        text = re.sub(source, target, text)

    #return strip_tags(text)
    return text

# num phrases = 30
# ngrams = (1, 3)

class InfoShieldCoarse():
    def __init__(self, filename, doc_text_header=None, doc_id_header=None, num_phrases=10):
        # init basic variables
        self.time = time.time()
        self.num_phrases = num_phrases
        self.filename_full = filename.split('.')[0]
        self.filename = os.path.basename(filename).split('.')[0]
        self.time_filename = '{}_streaming_time.txt'.format(self.filename)
        self.ngrams = (3, 5)
        self.index_to_docid = Counter()
        self.docid_to_index = Counter()

        self.data = pandas.read_csv(filename, lineterminator='\n')
        self.determine_header_names(doc_text_header, doc_id_header)

        self.num_ads = len(self.data.index)
        self.cluster_graph = nx.Graph()

        # setup tfidf - we want to keep emojis and capitalization
        tfidf = TfidfVectorizer(token_pattern=r'[^\s]+', lowercase=False, ngram_range=self.ngrams, sublinear_tf=True)
        self.tokenizer = tfidf.build_analyzer()
        self.data[self.description] = self.data.apply(lambda r: filter_text('{} {}'.format(r['title'], r[self.description])), axis=1)
        self.tfidfs = tfidf.fit_transform(self.data[self.description])
        self.tfidf_indices = tfidf.get_feature_names()
        print('done with tfidf', time.time() - self.time)


    def determine_header_names(self, doc_text_header, doc_id_header):
        ''' automatically determine relevant header names for doc id, doc text'''
        columns = set(self.data.columns)
        indices = {'ad_id', 'index', 'TweetID', 'id'}
        descriptions = {'u_Description', 'description', 'body', 'Tweet', 'text'}
        descriptions.add(doc_text_header)
        indices.add(doc_id_header)
        indices.add(doc_text_header)
        for name, field in [('text', descriptions), ('unique id', indices)]:
            if not len(columns.intersection(field)):
                print('Add "{}" header to possible descriptions!'.format(name))
                exit(1)
        self.description = columns.intersection(descriptions).pop()
        self.id = columns.intersection(indices).pop()


    def tokenize_text(self, text):
        return self.tokenizer(filter_text(text))


    def top_tfidf_phrases(self, index, return_all=False):
        ''' return the top phrases with highest tfidf score '''
        _, cols = self.tfidfs[index].nonzero()
        tfidf_pairs = [(self.tfidfs[index, c], self.tfidf_indices[c]) for c in cols]
        tfidf_pairs = sorted(tfidf_pairs, reverse=True)
        num_to_keep = self.num_phrases
        return heapq.nlargest(num_to_keep, tfidf_pairs)


    def process_ad(self, index, row):
        ''' find top phrases and add the ad to the cluster graph '''
        doc_id = row[self.id]
        self.index_to_docid[index] = doc_id
        self.docid_to_index[doc_id] = index

        t = time.time()
        top_tfidf = [phrase for _, phrase in self.top_tfidf_phrases(index)]

        self.cluster_graph.add_nodes_from(top_tfidf, bipartite=0)
        self.cluster_graph.add_node(doc_id, bipartite=1)
        self.cluster_graph.add_edges_from([(doc_id, phrase) for phrase in top_tfidf])

        #for field in ('title'):
        #    if field in self.data.columns and type(row[field]) == str:
        #        nodes = '{}-{}'.format(field, row[field].split(';'))
        #        self.cluster_graph.add_nodes_from(nodes, bipartite=0)
        #        self.cluster_graph.add_edges_from([(doc_id, n) for n in nodes])


    def generate_labels(self):
        document_nodes = set([n for n, d in self.cluster_graph.nodes(data=True) if d['bipartite']])
        for i, component in enumerate(nx.connected_components(self.cluster_graph)):
            docs = [c for c in component if c in document_nodes]
            if len(docs) < 2:
                continue

            self.data.loc[self.data.id.isin(docs), 'LSH label'] = i


    def write_cluster_graph(self):
        ''' write cluster graph as pkl file '''
        if not os.path.isdir('pkl_files'):
            os.mkdir('pkl_files')

        with open('pkl_files/{}_ad_graph.pkl'.format(self.filename), 'wb') as f:
            pickle.dump(self.cluster_graph, f)


    def write_csv_labels(self):
        ''' write new csv, with LSH labels '''
        self.final_data_filename = self.filename_full + '_LSH_labels.csv'
        self.unfiltered_data_filename = self.filename_full + '_full_LSH_labels.csv'

        data_filtered = self.data.dropna(subset=[self.description])
        data_filtered.to_csv(self.unfiltered_data_filename, index=False)
        data_filtered = data_filtered[data_filtered['LSH label'] != -1]
        data_filtered.to_csv(self.final_data_filename, index=False)


    def clustering(self):
        ''' process each ad individually and incrementally save cluster graph. '''
        t = time.time()

        for index, row in self.data.iterrows():
            # printing for sanity
            if index and not index % 10000:
                time_elapsed = time.time() - t
                print(index, '/', self.num_ads, 'time', time_elapsed)

            self.docid_to_index[row[self.id]] = index
            self.index_to_docid[index] = row[self.id]
            self.process_ad(index, row)

        self.generate_labels()
        self.write_cluster_graph()
        self.write_csv_labels()
        self.total_time = time.time() - t
        print('Finished clustering!', self.total_time)


    def process_batch(self, batch_num, deltas=True):
        ''' process each ad individually and incrementally save cluster graph. '''
        t = time.time()

        batch = self.data[self.data.batch_num == batch_num]
        for index, row in batch.iterrows():
            # printing for sanity
            if index and not index % 1000:
                time_elapsed = time.time() - t
                print(index, '/', self.num_ads, 'time', time_elapsed)

            self.docid_to_index[row[self.id]] = index
            self.index_to_docid[index] = row[self.id]
            self.process_ad(index, row)
            #print(time.time() - t)

        self.generate_labels()
        self.write_cluster_graph()

        if batch_num and deltas:
            # only pass new things through fine
            changed_clusters = set(df[self.data.batch_num == batch_num]['LSH label'].values)
            new_file = self.filename_full + 'LSH_batch.csv'
            self.data[self.data['LSH label'] == changed_clusters].to_csv(new_file, index=False)
        else:
            self.write_csv_labels()

        self.total_time = time.time() - t
        print('Finished batch {}!'.format(batch_num), self.total_time)


    def get_clusters(self):
        ''' given cluster graph, return the relevant connected components '''
        return [c for c in nx.connected_components(self.cluster_graph)]


    def get_docs(self):
        ''' return document nodes only '''
        return [n for n, d in self.cluster_graph.nodes(data=True) if d['bipartite']]


    def print_clusters(self):
        print('number of clusters', len(self.get_clusters()))
        clusters = self.get_clusters()
        document_nodes = self.get_docs()
        for i, cluster in enumerate(clusters):
            docs = [c for c in cluster if c in document_nodes]
            if len(docs) ==1:
                continue
            print('cluster:', i, 'len:', len(docs))
            for doc_id in docs:
                index = self.docid_to_index[doc_id]
                row = self.data.loc[index]
                try:
                    description = row[self.description]
                    print(doc_id, row[self.description], row['label'])
                except:
                    print('issue with doc_id', doc_id, 'and desc', self.description)
                print()
            print('\n\n')


def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage(1)

    filename = sys.argv[1]
    c = InfoShieldCoarse(filename)
    c.clustering()
