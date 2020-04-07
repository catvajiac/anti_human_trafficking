#!/usr/bin/env python3
# Author: Catalina Vajiac
# Purpose: catchall script to analyze output from streaming_alg.py
# Usage: ./analyze_streaming_alg.py [graph_pkl] [original_csv]

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import os, sys
import pandas
import pickle

from collections import Counter


def load_data(pkl_filename, data_filename):
    print('Loading graph...')
    graph = pickle.load(open(pkl_filename, 'rb'))
    data = pandas.read_csv(data_filename, encoding='utf-8')
    return data, graph


def plot_degree_distributions(graph):
    print('Finding degrees...')
    out_degree_counts = Counter()
    in_degree_counts = Counter()
    out_degree_list= []
    in_degree_list= []
    for node in graph.nodes:
        out_degree = sum([1 for _ in graph.neighbors(node)])
        in_degree = sum([1 for _ in graph.predecessors(node)])
        in_degree_counts[in_degree] += 1
        out_degree_counts[out_degree] += 1
        in_degree_list.append(in_degree)
        out_degree_list.append(out_degree)


    x = [key for key, value in in_degree_counts.items()]
    y = [value for key, value in in_degree_counts.items()]

    plt.scatter(x, y)
    plt.hist(in_degree_list, bins=50)
    plt.title('In-degree distribution')
    plt.xlabel('In-degree')
    plt.ylabel('Number of nodes')
    plt.show()

    x = [key for key, value in out_degree_counts.items()]
    y = [value for key, value in out_degree_counts.items()]

    plt.scatter(x, y)
    plt.hist(out_degree_list, bins=50)
    plt.title('Out-degree distribution')
    plt.xlabel('Out-degree')
    plt.ylabel('Number of nodes')
    plt.show()


def analyze_connected_components(graph, data):
    print('nodes', graph.number_of_nodes())
    print("Components:", nx.number_weakly_connected_components(graph))

    for i, nodes in enumerate(nx.weakly_connected_components(graph)):
        if len(nodes) < 10:
            continue
        for node in nodes:
            comp = graph.nodes[node]['contains']
            cluster = data.loc[data['ad_id'].isin(comp)]
            dups_shape = cluster.pivot_table(index=['u_Description'], aggfunc='size')
            print(cluster['u_Description'].iloc[0], '\n')
        print('\n\n')


def plot_posts_per_day(data):
    data['PostingDate'] = pandas.to_datetime(data['PostingDate'])
    groups = data.groupby(data['PostingDate'].dt.date).size()
    plt.title('Number of posts/day')
    groups.plot.bar(rot=80)
    plt.show()


if __name__ == '__main__':
    graph_pkl_filename = sys.argv[1]
    data_filename = sys.argv[2]
    data, graph = load_data(graph_pkl_filename, data_filename)

    #plot_degree_distributions(graph)
    #plot_posts_per_day(data)
    analyze_connected_components(graph, data)
