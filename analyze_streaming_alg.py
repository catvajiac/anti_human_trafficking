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
from functools import reduce

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
            if li[-1][1] >= item[0]:
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


def analyze_connected_components(graph, data):
    print('Nodes:', graph.number_of_nodes())
    print("Components:", nx.number_weakly_connected_components(graph), '\n')

    for i, nodes in enumerate(nx.weakly_connected_components(graph)):
        if len(nodes) < 2:
            continue
        for node in nodes:
            comp = graph.nodes[node]['contains']
            ad_text = data.data.loc[data.data['ad_id'].isin(comp)]['u_Description'].iloc[0]
            #ad_text = ad_text.replace('\', '\textbackslash ')
            ad_text = ad_text.replace('$', '')
            ad_text = ad_text.replace('{', '')
            ad_text = ad_text.replace('}', '')
            ad_text = ad_text.replace('(*@', '')
            ad_text = ad_text.replace('@*)', '')
            ad_text = ad_text.replace('&', '')
            ad_text = data.preprocess_ad(ad_text)
            intervals = [(start, end) for _, _, start, end in data.calc_tfidf(ad_text)]
            for start, end in merge(intervals)[::-1]:
                start_highlight = '(*@ \ctext{{'.format(COLOR)
                end_highlight = r'}@*)'
                ad_text.insert(start, start_highlight)
                ad_text.insert(end+1, end_highlight)
            print(len(comp))
            print(' '.join(ad_text))
            print('\n')
        print(r'(*@\newpage @*)')

    print(END_LATEX)


def plot_posts_per_day(data):
    data.data[DATE] = pandas.to_datetime(data.data[DATE])
    groups = data.data.groupby(data.data[DATE].dt.date).size()
    plt.title('Number of posts/day')
    groups.plot.bar(rot=80)
    plt.show()


if __name__ == '__main__':
    prefix = sys.argv[1]
    graph_pkl_filename = './pkl_files/{}_ad_graph.pkl'.format(prefix)
    data_csv_filename = './data/{}.csv'.format(prefix)
    print(BEGIN_LATEX)
    data, graph = load_data(graph_pkl_filename, data_csv_filename)

    #plot_degree_distributions(graph)
    #plot_posts_per_day(data)
    analyze_connected_components(graph, data)
