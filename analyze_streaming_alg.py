#!/usr/bin/env python3
# Author: Catalina Vajiac
# Purpose: catchall script to analyze output from streaming_alg.py
# Usage: ./analyze_streaming_alg.py [name of file]

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import os, sys
import pandas
import pickle

from collections import Counter
from functools import reduce
from sortedcontainers import SortedList

from streaming_alg import data

BEGIN_LATEX = r'''\documentclass[11pt]{article}
\usepackage{listings, xcolor, soul}
\lstset{
  basicstyle=\ttfamily,
  showstringspaces=false,
  breaklines=true,
  keywordstyle={\textit},
  escapeinside={(*@}{@*)}
}
\newcommand{\ctext}[1]{
  \begingroup
  \sethlcolor{cyan}%
  \hl{#1}%
  \endgroup
}
\begin{document}\begin{lstlisting}'''


END_LATEX = r'\end{lstlisting}\end{document}'

COLOR = 'cyan'
DATE = 'PostingDate'


def merge(inp):
    def merge_sub(li,item):
        if li:
            if li[-1][1] >= item[0] - 1:
                li[-1] = li[-1][0], max(li[-1][1],item[1])
                return li
        li.append(item)
        return li
    return reduce(merge_sub, sorted(inp), [])


def load_data(pkl_filename, data_filename):
    print('Loading graph...')
    graph = pickle.load(open(pkl_filename, 'rb'))
    csv_data = data(data_filename)
    return csv_data, graph


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


def remove_punctuation(text):
    for symbol in ['$', '{', '}', '(*@', '@*)', '&']:
        text.replace(symbol, '')
    return text


def highlight_ad_text(data, text):
    text = text.split()
    intervals = [(index, index+len(phrase)) for _, index, phrase in data.calc_tfidf(text)]
    for start, end in merge(intervals):
        print(start, end)
        start_highlight = '(*@ \ctext{{'.format(COLOR)
        end_highlight = r'}@*)'
        text.insert(start, start_highlight)
        text.insert(end+1, end_highlight)

    return ' '.join(text)


def analyze_connected_components(graph, data):
    print('Nodes:', graph.number_of_nodes())
    print("Components:", nx.number_weakly_connected_components(graph), '\n')

    for i, nodes in enumerate(nx.weakly_connected_components(graph)):
        for node in nodes:
            comp = graph.nodes[node]['contains']
            ad_texts = data.data.loc[data.data['ad_id'].isin(comp)]['u_Description']
            print(len(ad_texts))
            for text in ad_texts:
                text = highlight_ad_text(data, text)
                print(text, '\n')
        print(r'(*@\newpage @*)')


def plot_posts_per_day(data):
    data.data[DATE] = pandas.to_datetime(data.data[DATE])
    groups = data.data.groupby(data.data[DATE].dt.date).size()
    plt.title('Number of posts/day')
    groups.plot.bar(rot=80)
    plt.show()


def plot_phone_number(graph, data):
    suspicious = []
    phones = SortedList()
    for i, nodes in enumerate(nx.weakly_connected_components(graph)):
        num_phones = 0
        for node in nodes:
            cluster_node = graph.nodes[node]['contains']
            ads = data.data.loc[data.data['ad_id'].isin(cluster_node)]
            num_phones += ads['e_PhoneNumbers'].count()

        phones.add(num_phones)
        if num_phones > 100:
            suspicious.append(i)

    plt.scatter(list(range(len(phones))), phones)
    plt.savefig('plot_phone.png')
    return suspicious


def peek_clusters(graph, data, components):
    for i, component in enumerate(nx.weakly_connected_components(graph)):
        if i not in components:
            continue

        for cluster_node in component:
            ad_ids = graph.nodes[cluster_node]['contains']
            ad_texts = data.data.loc[data.data['ad_id'].isin(ad_ids)]['u_Description']

            for text in ad_texts:
                print(text)
                print('\n')

        print(r'(*@\newpage @*)')


if __name__ == '__main__':
    prefix = sys.argv[1]
    graph_pkl_filename = './pkl_files/{}_ad_graph.pkl'.format(prefix)
    data_csv_filename = './data/{}.csv'.format(prefix)
    print(BEGIN_LATEX)
    data, graph = load_data(graph_pkl_filename, data_csv_filename)

    #plot_degree_distributions(graph)
    #plot_posts_per_day(data)
    data.find_idf()
    #sus = plot_phone_number(graph, data)
    analyze_connected_components(graph, data)
    #peek_clusters(graph, data, sus)
    print(END_LATEX)
