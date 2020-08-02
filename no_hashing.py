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

import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from datetime import datetime
from networkx.algorithms import bipartite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score, normalized_mutual_info_score


def term_frequency(phrase, document):
    return sum([p == phrase for p in document])


def filter_text(text):
    if type(text) is not str: # nan
        return ''

    replace = [(r'\d+', ''), (r'[^\x00-\x7F]+', ' '), (re.compile('<.*?>'), '')]
    for source, target in replace:
        text = re.sub(source, target, text)

    return text


class AutoDupCoarse():
    def __init__(self, filename, doc_text_header=None, doc_id_header=None, mode='normal', num_phrases=5):
        # init basic variables
        self.mode = mode
        self.num_phrases = num_phrases
        self.filename_full = filename.split('.')[0]
        self.filename = os.path.basename(filename).split('.')[0]
        self.time_filename = '{}_streaming-{}_time.txt'.format(self.filename, self.mode)
        self.ngrams = (5, 5)
        self.time = 0
        self.index_to_docid = Counter()
        self.docid_to_index = Counter()

        self.data = pandas.read_csv(filename, lineterminator='\n')
        #self.data = self.data.drop_duplicates(subset='ad_id', keep="first")

        self.determine_header_names(doc_text_header, doc_id_header)

        if 'timestamp' in self.data.columns:
            self.data.sort_values(by=['timestamp']) # since ads not in order as they should be
        self.num_ads = len(self.data.index)
        self.cluster_graph = nx.Graph()

        # setup tfidf
        self.tokenizer = TfidfVectorizer(lowercase=True, ngram_range=self.ngrams).build_analyzer()
        self.data[self.description] = self.data[self.description].apply(filter_text)
        self.document_frequency = defaultdict(float)


    def determine_header_names(self, doc_text_header, doc_id_header):
        ''' automatically determine relevant header names for doc id, doc text'''
        columns = set(self.data.columns)
        indices = {'ad_id', 'index', 'TweetID', 'id'}
        descriptions = {'u_Description', 'description', 'body', 'Tweet', 'text'}
        phones = {'u_PhoneNumbers', 'phone', 'PhoneNumber'}
        descriptions.add(doc_text_header)
        indices.add(doc_id_header)
        indices.add(doc_text_header)
        for name, field in [('text', descriptions), ('unique id', indices), ('phone #', phones)]:
            if not len(columns.intersection(field)):
                print('Add "{}" header to possible descriptions!'.format(name))
                exit(1)
        self.description = columns.intersection(descriptions).pop()
        self.id = columns.intersection(indices).pop()
        self.phone = columns.intersection(phones).pop()


    def tokenize_text(self, row):
        include_field = lambda x: x in self.data.columns and type(row[x]) == str
        def get_fields(prefix, field):
            if not include_field(field):
                return []
            return ['{}{}'.format(prefix, num) for num in row[field].split(';')]

        phrases = self.tokenizer(row[self.description])
        fields = (self.description, 'title', 'social', 'email')

        phones = get_fields('#', self.phone)
        images = get_fields('img', 'image_id')

        return phrases + phones + images



    def update_doc_freq(self, df):
        ''' for normal and batch methods: update document frequency given a dataframe '''
        if self.mode == 'stream':
            return

        for _, row in df.iterrows():
            phrases = self.tokenize_text(row)
            for phrase in set(phrases):
                self.document_frequency[phrase] += 1


    def top_tfidf_phrases(self, index, phrases, return_all=False):
        ''' return the top phrases with highest tfidf score '''
        def stream_tfidf(phrase, document, N):
            return term_frequency(phrase, document) * math.log((N+1)/self.document_frequency[phrase])

        if self.mode == 'stream':
            for phrase in set(phrases):
                self.document_frequency[phrase] += 1

        tfidf_pairs = [(stream_tfidf(phrase, phrases, index), phrase) for phrase in phrases]
        return heapq.nlargest(self.num_phrases, tfidf_pairs)


    def process_ad(self, index, row):
        ''' find top phrases and add the ad to the cluster graph '''
        doc_id = row[self.id]
        self.index_to_docid[index] = doc_id
        self.docid_to_index[doc_id] = index

        top_tfidf = [phrase for _, phrase in self.top_tfidf_phrases(index, self.tokenize_text(row))]
        self.cluster_graph.add_nodes_from(top_tfidf, bipartite=0)
        self.cluster_graph.add_node(doc_id, bipartite=1)
        self.cluster_graph.add_edges_from([(doc_id, phrase) for phrase in top_tfidf])

        '''
        if type(row[self.phone]) != str: # nan
            return
        phones = ['#{}'.format(phone) for phone in row[self.phone].split(';')]
        self.cluster_graph.add_nodes_from(phones, bipartite=0)
        self.cluster_graph.add_edges_from([(phone, doc_id) for phone in phones])
        '''

    def generate_labels(self):
        document_nodes = set([n for n, d in self.cluster_graph.nodes(data=True) if d['bipartite']])
        self.labels = [-1]*len(document_nodes)
        for i, component in enumerate(nx.connected_components(self.cluster_graph)):
            docs = [c for c in component if c in document_nodes]
            if len(docs) < 5:
                continue
            if not i % 5000:
                print(i)
            for docid in docs:
                self.labels[self.docid_to_index[docid]] = i

        pickle.dump(self.labels, open('labels.pkl', 'wb'))


    def write_cluster_graph(self):
        ''' write cluster graph as pkl file '''
        if not os.path.isdir('pkl_files'):
            os.mkdir('pkl_files')

        with open('pkl_files/{}_ad_graph.pkl'.format(self.filename), 'wb') as f:
            pickle.dump(self.cluster_graph, f)


    def write_csv_labels(self):
        ''' write new csv, with LSH labels '''
        self.final_data_filename = self.filename_full + '_LSH_labels.csv'

        self.data['LSH label'] = self.labels
        data_filtered = self.data.dropna(subset=[self.description])
        is_keep = lambda x: len(self.tokenizer(x)) >= 5
        data_filtered = data_filtered[data_filtered[self.description].map(is_keep)]
        data_filtered = data_filtered[data_filtered['LSH label'] != -1]
        data_filtered.to_csv(self.final_data_filename)


    def get_batches(self):
        # makes batches of dataframe
        tenth = math.ceil(len(self.data.index) / 10)
        position = 3*tenth
        yield self.data[:position]
        while position < len(self.data.index):
            yield self.data[position:min(position+tenth, len(self.data.index))]
            position += tenth

        if position != len(self.data.index):
            yield self.data[position:]


    def clustering(self):
        ''' process each ad individually and incrementally save cluster graph. '''
        t = time.time()
        # batch will have > 1 element only if self.mode is 'batch'
        batches = self.get_batches() if self.mode == 'batch' else [self.data]
        batch_numbers = [0]*len(self.data.index)

        index = 0
        for batch_num, batch in enumerate(batches):
            self.update_doc_freq(batch)
            for _, row in batch.iterrows():
                if index and not index % 10000:
                    time_elapsed = time.time() - t
                    print(index, '/', self.num_ads, 'time', time_elapsed)
                    #print('\t', self.time)

                batch_numbers[index] = batch_num
                self.docid_to_index[row[self.id]] = index
                self.index_to_docid[index] = row[self.id]
                self.process_ad(index, row)
                index += 1

        self.data['batch_num'] = batch_numbers
        self.generate_labels()
        self.write_cluster_graph()
        self.write_csv_labels()
        self.total_time = time.time() - t
        print('Finished clustering!', self.total_time)
        with open('TIMES.txt', 'a') as f:
            f.write('{} {}\n'.format(self.filename, self.total_time))


    def get_clusters(self):
        ''' given cluster graph, return the relevant connected components '''
        criteria = lambda x: len(x) > 1 or len(self.get_docs(x)) > 1
        return [c for c in nx.connected_components(self.cluster_graph) if criteria(c)]


    def get_docs(self, cluster_nodes):
        ''' given a set of cluster nodes, return the documents they represent '''
        return cluster_nodes


    def print_clusters(self):
        print('number of clusters', len(self.get_clusters()))
        clusters = sorted(self.get_clusters(), key=lambda x: len(self.get_docs(x)), reverse=True)
        document_nodes = [n for n, d in self.cluster_graph.nodes(data=True) if d['bipartite']]
        for i, cluster in enumerate(clusters):
            docs = [c for c in cluster if c in document_nodes]
            if len(docs) < 5:
                continue
            print('cluster:', i, 'len:', len(docs))
            for doc_id in docs:
                index = self.docid_to_index[doc_id]
                row = self.data.loc[index]
                try:
                    description = row[self.description]
                except:
                    print('issue with doc_id', doc_id, 'and desc', self.description)
                print(doc_id, description)
                print()
            print('\n\n')


def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


def barplot(data1, data2):
    c1 = Counter()
    for d in data1:
        c1[d] += 1

    c2 = Counter()
    for d in data2:
        c2[d] += 1

    allkeys = list(c1.keys()) + list(c2.keys())

    bins = np.linspace(min(allkeys), max(allkeys), 100)
    plt.hist(data1, bins, alpha=0.5, label='Normal tfidf')
    plt.hist(data2, bins, alpha=0.5, label='Streaming tfidf')
    plt.yscale('log')
    plt.legend()
    plt.show()


def post_process(data):
    document_nodes = [n for n, d in data.cluster_graph.nodes(data=True) if d['bipartite'] == 1]
    #graph = bipartite.projected_graph(data.cluster_graph, document_nodes)
    #non_singleton = [len(comp) for comp in nx.connected_components(graph) if len(comp) > 1]

    #print('num components:', nx.number_connected_components(graph))
    #print('num non-singleton:', len(non_singleton))
    #print('max component:', max(non_singleton))
    #print('min component:', min(non_singleton))
    #print('avg component:', sum(non_singleton) / len(non_singleton))
    #print('nodes:', graph.number_of_nodes())
#
    return data.labels, data.cluster_graph


def stream_experiment(filename):
    for mode in ('normal', 'stream', 'batch'):
        d = AutoDupCoarse(filename, mode=mode, num_phrases=5)
        d.clustering()
        labels = d.labels
        true_labels = d.data['cluster_label'].values

        print('\n{} stats: with noise in one cluster'.format(mode))
        print('ari', adjusted_rand_score(true_labels, labels))
        print('hom', homogeneity_score(true_labels, labels))
        print('nmi', normalized_mutual_info_score(true_labels, labels))

        indices = [i for i, label in enumerate(true_labels) if label != -1]
        true_labels_noise = [label if i in incides else i for i, label in enumerate(true_labels)]
        labels_noise = [label if i in indices else i in i for i, label in enumerate(labels)]
        print('\n{} stats: with noise in all cluster'.format(mode))
        print('ari', adjusted_rand_score(true_labels_noise, labels))
        print('hom', homogeneity_score(true_labels_noise, labels))
        print('nmi', normalized_mutual_info_score(true_labels_noise, labels))

        true_labels_no_noise = [label for i, label in enumerate(true_labels) if i in indices]
        labels_no_noise = [label for i, label in enumerate(labels) if i in indices]
        print('\n{} stats: without noise'.format(mode))
        print('ari', adjusted_rand_score(true_labels_no_noise, labels_no_noise))
        print('hom', homogeneity_score(true_labels_no_noise, labels_no_noise))
        print('nmi', normalized_mutual_info_score(true_labels_no_noise, labels_no_noise))




def basic_stats(filename, mode='normal', num_phrases=5):
    d = AutoDupCoarse(filename, mode=mode, num_phrases=num_phrases)
    d.clustering()
    #d.print_clusters()

    #label_list, graph = post_process(d)
    #non_singleton = [len(comp) for comp in nx.connected_components(graph) if len(comp) > 1]
    true_labels = d.data['cluster_label'].values
    labels = d.labels
    print('\n{} stats: with noise in one cluster'.format(mode))
    print('ari', adjusted_rand_score(true_labels, labels))
    print('hom', homogeneity_score(true_labels, labels))
    print('nmi', normalized_mutual_info_score(true_labels, labels))

    indices = [i for i, label in enumerate(true_labels) if label != -1]
    true_labels_noise = [label if i in indices else i for i, label in enumerate(true_labels)]
    labels_noise = [label if i in indices else i for i, label in enumerate(labels)]
    print('\n{} stats: with noise in all cluster'.format(mode))
    print('ari', adjusted_rand_score(true_labels_noise, labels_noise))
    print('hom', homogeneity_score(true_labels_noise, labels_noise))
    print('nmi', normalized_mutual_info_score(true_labels_noise, labels_noise))

    true_labels_no_noise = [label for i, label in enumerate(true_labels) if i in indices]
    labels_no_noise = [label for i, label in enumerate(labels) if i in indices]
    print('\n{} stats: without noise'.format(mode))
    print('ari', adjusted_rand_score(true_labels_no_noise, labels_no_noise))
    print('hom', homogeneity_score(true_labels_no_noise, labels_no_noise))
    print('nmi', normalized_mutual_info_score(true_labels_no_noise, labels_no_noise))

    '''
    print('num components:', nx.number_connected_components(graph))
    print('num non-singleton:', len(non_singleton))
    print('max component:', max(non_singleton))
    print('min component:', min(non_singleton))
    print('avg component:', sum(non_singleton) / len(non_singleton))
    print('nodes:', graph.number_of_nodes())
    '''
    num_components = nx.number_connected_components(graph)
    print('num_components', num_components)

    return num_components, len(non_singleton), max(non_singleton), sum(non_singleton) / len(non_singleton), d


def one_experiment(filename, num_phrases, mode):
    d = AutoDupCoarse(filename, mode=mode, num_phrases=num_phrases)
    d.clustering()
    return d.labels, d.data['cluster_label'].values


def all_experiments(filename):
    top_num = 11
    normal_stats = [one_experiment(filename, num_phrases=i, mode='normal') for i in range(1, top_num)]
    stream_stats = [one_experiment(filename, num_phrases=i, mode='stream') for i in range(1, top_num)]
    batch_stats = [one_experiment(filename, num_phrases=i, mode='batch') for i in range(1, top_num)]

    x = list(range(1, top_num))
    for name, metric in [('ARI', adjusted_rand_score), ('Homogeneity', homogeneity_score), ('NMI', normalized_mutual_info_score)]:
        plt.plot(x, [metric(*tup) for tup in normal_stats], label='Normal tfidf')
        plt.plot(x, [metric(*tup) for tup in stream_stats], label='Stream tfidf')
        plt.plot(x, [metric(*tup) for tup in batch_stats], label='Batch tfidf')
        plt.xlabel('# phrases extracted')
        plt.xticks(rotation=70, fontsize=10)
        plt.ylabel(name)
        plt.legend()
        plt.savefig('num_phrases-{}.png'.format(name.replace(' ', ')')))
        plt.clf()


def phrases_experiment(filename):
    top_num = 11
    normal_stats = [basic_stats(filename, num_phrases=i, mode='normal') for i in range(1, top_num)]
    stream_stats = [basic_stats(filename, num_phrases=i, mode='stream') for i in range(1, top_num)]
    batch_stats = [basic_stats(filename, num_phrases=i, mode='batch') for i in range(1, top_num)]

    normal_label_lists = [tup[4].label_list for tup in normal_stats]
    stream_label_lists = [tup[4].label_list for tup in stream_stats]
    batch_label_lists = [tup[4].label_list for tup in batch_stats]
    true_labels = normal_stats[0][-1].labels


    x = list(range(1, top_num))
    '''
    for i, name in enumerate(('# clusters', '# non-singleton clusters', 'Max cluster size', 'Avg cluster size')):
        plt.plot(x, [tup[i] for tup in normal_stats], label='Normal tfidf')
        plt.plot(x, [tup[i] for tup in stream_stats], label='Stream tfidf')
        plt.plot(x, [tup[i] for tup in batch_stats], label='Batch tfidf')
        plt.xlabel('Number of phrases extracted')
        plt.ylabel(name)
        plt.legend()
        plt.savefig('num_phrases-{}.png'.format(name.replace(' ', '_')))
        plt.clf()
    '''

    label_x = ['{}-{}'.format(a, b) for a, b in zip(x, x[1:])]
    get_pairs = lambda x: zip(x, true_labels)
    for name, metric in [('ARI', adjusted_rand_score), ('Homogeneity', homogeneity_score), ('NMI', normalized_mutual_info_score)]:
        plt.plot(label_x, [metric(data, true_labels) for data in normal_label_lists], label='Normal tfidf')
        plt.plot(label_x, [metric(data, true_labels) for data in stream_label_lists], label='Stream tfidf')
        plt.plot(label_x, [metric(data, true_labels) for data in batch_label_lists], label='Batch tfidf')
        plt.xlabel('# phrases extracted')
        plt.xticks(rotation=70, fontsize=10)
        plt.ylabel(name)
        plt.legend()
        plt.savefig('num_phrases-{}.png'.format(name.replace(' ', ')')))
        plt.clf()


if __name__ == '__main__':
    # either just provide filename, or provide all params
    if len(sys.argv) not in [2, 5]:
        usage(1)

    filename = sys.argv[1]
    basic_stats(filename, mode='normal')


    #phrases_experiment(filename)
    #all_experiments(filename)
    #stream_experiment(filename)
