#!/usr/bin/env python3
# Author:   Catalina Vajiac
# Purpose:  Use LSH idea to cluster ad data
# Usage:    ./streaming_alg.py [filename]

import re
import math
import networkx as nx
import numpy as np
import os, sys
import pandas
import pickle
import random
import time

from collections import Counter, defaultdict
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations


class hash_family():
    def __init__(self, num_hash_functions, buckets_in_hash):
        self.num_hash_functions = num_hash_functions
        self.hash_cutoff = num_hash_functions / 2
        self.duplicate_cutoff = num_hash_functions * 3 / 4
        self.buckets_in_hash = buckets_in_hash

        random_generator = lambda: random.randrange(buckets_in_hash)
        self.hash_functions = [defaultdict(random_generator) for _ in range(num_hash_functions)]
        self.hash_tables = [defaultdict(list) for _ in range(num_hash_functions)]
        self.marked = set()
        self.max_bucket = 0


    def get_hashes(self):
        ''' return pairs of hash functions and hash tables '''
        for h, table in zip(self.hash_functions, self.hash_tables):
            yield h, table

    def add_to_hash_tables(self, to_hash, to_add):
        ''' adds to_add in place that to_hash hashes, for all hash functions '''
        for h, table in zip(self.hash_functions, self.hash_tables):
            table[h[to_hash]] = list(set([val for val in table[h[to_hash]] if val not in self.marked]))
            if to_add not in table[h[to_hash]]:
                table[h[to_hash]].append(to_add)

            if len(table[h[to_hash]]) > 100:
                table[h[to_hash]].pop(0)

            self.max_bucket = max(self.max_bucket, len(table[h[to_hash]]))


    def __repr__(self):
        ''' prints all nonzero buckets, with elements, for each hash table '''
        for index, table in enumerate(self.hash_tables):
            print('Table', index)
            for bucket, elements in table.items():
                print('  ', bucket, ':', ' '.join(map(str, elements)))



class data():
    def __init__(self, filename, num_hash_functions=128, num_buckets=400, ngrams=[2, 3]):
        self.filename_full = filename.split('.')[0]
        self.filename = os.path.basename(filename).split('.')[0]
        self.time_filename = self.filename + '_time.txt'
        self.ngrams = ngrams
        self.time = 0

        self.data = pandas.read_csv(filename)

        # automatically determine relevant header names
        descriptions = {'u_Description', 'description', 'body'}
        indices = {'ad_id', 'index'}
        self.description = set(self.data.columns).intersection(descriptions).pop()
        self.ad_id = set(self.data.columns).intersection(indices).pop()

        if 'PostingDate' in self.data.columns:
            self.data.sort_values(by=['PostingDate']) # since ads not in order as they should be
        self.num_ads = len(self.data.index)
        self.cluster_graph = nx.DiGraph()
        self.hashes = hash_family(num_hash_functions, num_buckets)


    def remove_html(self, text):
        if type(text) is not str: # nan
            return ''

        replace = [(r'\d+', ''), (r'[^\x00-\x7F]+', ' '), (re.compile('<.*?>'), '')]
        for source, target in replace:
            text = re.sub(source, target, text)

        return text


    def preprocess_ad(self, ad):
        return word_tokenize(self.remove_html(ad))


    def get_tokens(self, ad_text, give_index=False):
        ad_text = self.preprocess_ad(ad_text)
        for ngram in self.ngrams:
            for index, token in enumerate(list(zip(*[ad_text[i:] for i in range(ngram)]))):
                if give_index:
                    yield index, token
                else:
                    yield token


    def find_tfidf(self):
        ''' pre-calculate tfidf '''
        print('Finding tfidf...')
        stop_words = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(lowercase=False, ngram_range=self.ngrams, norm='l2',
                smooth_idf=True, stop_words=stop_words, min_df=2, max_df=0.8)
        data = self.data[self.description].apply(self.remove_html)
        self.tfidf = vectorizer.fit_transform(data)
        self.tfidf_indices = vectorizer.get_feature_names()
        self.tokenizer = vectorizer.build_tokenizer()


    def calc_tfidf(self, index, return_all=False):
        ''' return the top phrases with highest tfidf score '''
        ad_tokens = self.preprocess_ad(self.data.loc[index][self.description])

        doc_tfidf = self.tfidf[index]
        _, indices = doc_tfidf.nonzero()

        tfidf_sorted = sorted([(self.tfidf[index, i], self.tfidf_indices[i]) for i in indices], reverse=True)
        total_len = 0
        num_phrases = 1

        # determine number of phrases to return: want 50% of ad length
        while total_len < math.ceil(len(ad_tokens)) and num_phrases < len(tfidf_sorted):
            total_len += len(tfidf_sorted[num_phrases][1])
            num_phrases += 1

        return tfidf_sorted[:num_phrases]


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
            if all([count >= self.hashes.hash_cutoff for _, count in phrase_count.items()]):
                return {}, 'duplicate', cluster

            if all([count <= self.hashes.hash_cutoff for _, count in phrase_count.items()]):
                related_clusters.pop(cluster)

        self.time += (time.time() - t)
        cluster_id = self.cluster_graph.number_of_nodes()
        self.hashes.marked.update(related_clusters)
        return related_clusters, 'chain', cluster_id


    def add_new_cluster(self, related_clusters, cluster_id, ad_id):
        ''' add cluster to cluster_graph, with related edges '''
        self.cluster_graph.add_node(cluster_id, contains=list([ad_id]))
        self.cluster_graph.add_edges_from([(rel, cluster_id) for rel in related_clusters])


    def process_ad(self, index, row):
        ''' find top phrases, use them to find related clusters, and add the ad to the cluster
        graph '''
        ad_text = self.preprocess_ad(row[self.description])
        ad_id = row[self.ad_id]

        top_tfidf = [phrase for _, phrase in self.calc_tfidf(index)]
        related_clusters, cluster_type, cluster_id = self.find_related_clusters(top_tfidf, ad_id)

        if cluster_type == 'duplicate':
            self.cluster_graph.nodes[cluster_id]['contains'].append(ad_id)
            return

        self.add_new_cluster(related_clusters, cluster_id, ad_id)

        for phrase in top_tfidf:
            self.hashes.add_to_hash_tables(phrase, cluster_id)


    def write_cluster_graph(self):
        ''' write cluster graph as pkl file '''
        with open('pkl_files/{}_ad_graph.pkl'.format(self.filename), 'wb') as f:
            pickle.dump(self.cluster_graph, f)


    def write_csv_labels(self):
        ''' write new csv, with LSH labels '''
        my_labels = [0]*len(self.data.index)
        for i, cluster in enumerate(nx.weakly_connected_components(self.cluster_graph)):
            for ad in self.get_ads(cluster):
                my_labels[ad] = i
        self.data['LSH label'] = my_labels
        self.data.to_csv(self.filename_full + '_LSH_labels.csv')


    def clustering(self):
        ''' process each ad individually and incrementally save cluster graph. '''
        t = time.time()
        self.find_tfidf()
        print('Finished tfidf in time:', time.time() - t)
        print('Starting clustering...')

        # assume in order of timestamp (streaming case)
        for index, row in self.data.iterrows():
            if index and not index % 1000:
                time_elapsed = time.time() - t
                print(index, '/', self.num_ads, 'time', time_elapsed)
                print('\t', self.time)
                print('\t', 'max bucket size', self.hashes.max_bucket)
                self.write_cluster_graph()
                with open(self.time_filename, 'a') as f:
                    f.write('{} {}\n'.format(index, time_elapsed))

            self.process_ad(index, row)

        self.write_cluster_graph()
        self.write_csv_labels()
        self.total_time = time.time() - t
        print('Finished clustering!', time.time() - t)


    def visualize_buckets(self):
        ''' creates visual representation of how full the buckets are for a hash table '''
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


    def get_clusters(self):
        ''' given cluster graph, return the relevant connected components '''
        criteria = lambda x: len(x) > 1 or len(self.get_ads(x)) > 1
        return [c for c in nx.weakly_connected_components(self.cluster_graph) if criteria(c)]


    def get_ads(self, cluster_nodes):
        ''' given a set of cluster nodes, return the ads they represent '''
        return [ad for node in cluster_nodes for ad in self.cluster_graph.nodes[node]['contains']]


    def print_clusters(self):
        print('number of clusters', len(self.get_clusters()))
        for i, cluster in enumerate(self.get_clusters()):
            print('cluster', i)
            print('contains cluster_ids', cluster)
            print(list(self.get_ads(cluster)))
            for ad_id in self.get_ads(cluster):
                try:
                    description = self.remove_html(self.data.loc[ad_id][self.description])
                except:
                    print('issue with ad_id', ad_id, 'and desc', self.description)
                true_label = self.data.loc[ad_id]['label']
                print(true_label, description)
                print()
            print('\n\n')


    def compare_true_synthetic(self):
        true_labels = self.data['label'].values

        my_labels = [0]*len(self.data.index)
        for i, cluster in enumerate(nx.weakly_connected_components(self.cluster_graph)):
            for ad in self.get_ads(cluster):
                my_labels[ad] = i

        nmi = normalized_mutual_info_score(true_labels, my_labels)
        hom = homogeneity_score(true_labels, my_labels)
        ari = adjusted_rand_score(true_labels, my_labels)
        print('NMI        ', nmi)
        print('HOMOGENEITY', hom)
        print('RAND INDEX ', ari)
        return nmi, hom, ari


def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    #filename = sys.argv[1]
    #canadian_data = data(filename)
    #name = 'pkl_files/{}_ad_graph.pkl'.format(canadian_data.filename)
    #run_parameter_experiment()
    filename = sys.argv[1]
    hf, nb, n1, n2 = map(int, sys.argv[2:])
    d = data(filename, num_hash_functions=hf, num_buckets=nb, ngrams=(n1, n2))
    d.clustering()
    score = d.compare_true_synthetic()
    #d.print_clusters()
    experiment_filename = os.path.basename(filename).split('.')[0] + '_paramTFIDF.txt'
    with open(experiment_filename, 'a') as f:
        f.write('{} {} {} {} {} {} {} {}\n'.format(hf, nb, n1, n2, d.total_time, *score))


    #canadian_data.cluster_graph = pickle.load(open(name, 'rb'))
    #canadian_data.clustering()
    #canadian_data.compare_true_synthetic()
