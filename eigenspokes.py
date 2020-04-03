#!/usr/bin/env python3
# Author:   Catalina Vajiac
# Purpose:  decides which eigenvectors store ''useful'' information
# Usage:    ./eigenspokes.py [filename]

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os, sys
import pickle

from collections import defaultdict
from itertools import product
from sortedcontainers import SortedList
from scipy.stats import multivariate_normal
import scipy as sp


# global vars

ZERO_CUTOFF = 1e-2
PLOTS_PATH = './plots/eigenspokes/'
PKL_PATH = './pkl_files'
PATHS = [PLOTS_PATH, PKL_PATH]


# Utility functions

def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


def read_data(filename):
    ''' Assume data format is edge list split by newlines, autoconverts labels to integers '''
    if filename.endswith('.pkl'):
        return pickle.load(open(filename, 'rb'))
    with open(filename, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return nx.convert_node_labels_to_integers(graph)


def modularity(subgraph, graph):
    ''' returns number of communities found '''
    return len(nx.algorithms.community.greedy_modularity_communities(graph)) > 1

def conductance(subgraph, graph):
    #print(nx.algorithms.cuts.conductance(graph, subgraph.nodes))
    return nx.algorithms.cuts.conductance(graph, subgraph.nodes) < 0.93


def filter_zeros(u_x, u_y):
    ''' removes pairs of x and y that are close to zero '''
    plot_x, plot_y = [], []
    for x, y in zip(u_x, u_y):
        if abs(x) <= ZERO_CUTOFF and abs(y) <= ZERO_CUTOFF:
            continue
        plot_x.append(x)
        plot_y.append(y)

    return plot_x, plot_y


def calc_entropy(x, y):
    ''' fits a 2D gaussian to point cloud and calculates the entropy of the data given the
        gaussian model '''
    # note: have to add noise to make cov matrix not singular
    data = np.stack((x, y), axis=0)# + .001*np.random.rand(2, len(x))
    cov = np.cov(data)
    mean = [sum(x) / len(x), sum(y) / len(y)]
    try:
        entropy = abs(multivariate_normal(mean=mean, cov=cov).entropy())
    except:
        entropy = float('inf')

    return entropy


# Core functions

def svd(graph, filename, use_pkl=True):
    ''' returns tuple: (u, s, v) '''
    svd_pkl_filename = '{}/{}_svd.pkl'.format(PKL_PATH, filename)
    if use_pkl and os.path.exists(svd_pkl_filename):
        print('Using SVD pkl file...')
        return pickle.load(open(svd_pkl_filename, 'rb'))

    print('Running SVD...')
    #array = nx.to_numpy_matrix(graph)
    array = nx.to_scipy_sparse_matrix(graph, dtype='float64')
    u, s, v = sp.sparse.linalg.svds(array, k=9, which='LM')
    u = np.array(u)
    if use_pkl:
        pickle.dump((u, s, v), open(svd_pkl_filename, 'wb'))

    return u, s, v


def find_spoke(scores, graph, metric=conductance):
    ''' first: find the point with highest projection, that is current spoke/subgraph
        then: always process a neighboring node to current subgraph which has the highest proj
        stop: when modularity metric says there are multiple communities '''
    nodes_to_scores = {i: x for i, x in scores}
    visited = set()

    key = lambda tup: tup[1]
    i, x = max(scores, key=key)
    neighbors = SortedList([(n, nodes_to_scores[n]) for n in  graph.neighbors(i)], key=key)
    spoke = set([i])

    while len(neighbors):
        i, x = neighbors.pop(-1)
        visited.add(i)
        if x <= ZERO_CUTOFF:
            continue
        spoke.add(i)

        spoke_graph = graph.subgraph(spoke)
        if metric(spoke_graph, graph):
            spoke.remove(i)
            break

        for neighbor in graph.neighbors(i):
            if neighbor in visited:
                continue
            neighbors.add((neighbor, nodes_to_scores[neighbor]))

    return list(spoke)


def find_spokes(graph, u, filename, use_pkl=True):
    ''' Find spokes - for now just looks at top 9 eigenvectors for ease of plotting later '''
    spokes_pkl_filename = '{}/{}_spokes.pkl'.format(PKL_PATH, filename)
    if use_pkl and os.path.exists(spokes_pkl_filename):
        print('Using spokes pkl file...')
        return pickle.load(open(spokes_pkl_filename, 'rb'))

    print('Finding spokes...')
    spokes = []
    for axis in range(9):
        if u.shape[0] <= axis:
            break
        scores = [(i, abs(x)) for i, x in enumerate(u[:, axis])]
        spokes.append(find_spoke(scores, graph))
        print('  u{} spoke found. size: {}'.format(axis, len(spokes[-1])))

    if use_pkl:
        pickle.dump(spokes, open(spokes_pkl_filename, 'wb'))
    return spokes


def refine_pairwise_spokes(u, spokes):
    ''' only keeps spokes that create useful EE plots '''
    data = {(x, y): defaultdict(list) for x, y in product(range(9), repeat=2)}

    for x_axis in range(9):
        for y_axis in range(9):
            if u.shape[0] <= x_axis or u.shape[0] <= y_axis:
                data.pop((x_axis, y_axis))
                continue
            x, y = filter_zeros(u[:, x_axis], u[:, y_axis])
            data[x_axis, y_axis]['points'] = (x, y)

            for direction in [x_axis, y_axis]:
                spoke = spokes[direction]
                x = u[spoke, x_axis]
                y = u[spoke, y_axis]
                if not len(x) or calc_entropy(x, y) < 5:
                    continue
                data[x_axis, y_axis]['spoke'].append((x, y))

    return data


def filter_singular_vectors(data):
    ''' this returns a set of singular vectors that create spokes in EE plots with all other
        singular vectors in the set '''
    max_sv = max([max(x, y) for x, y in data]) + 1
    singular_vectors = set()
    for x in range(9):
        if sum(([any(data[x, y]['spoke']) for y in range(max_sv)])) == 9:
            singular_vectors |= set([x])
            continue
        spokes = set([y for y in range(max_sv) if y != x and len(data[x, y]['spoke'])])
        singular_vectors |= spokes

    return singular_vectors


def plot_spokes(data):
    ''' plots EE plots of the first 9 eigenvectors '''

    f, ((a1, a2, a3), (a4, a5, a6), (a7, a8, a9))= plt.subplots(3, 3,  sharex='col', sharey='row', figsize=(8,8), dpi=80)

    axes = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

    save_path = '{}/{}'.format(PLOTS_PATH, filename)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for x_axis in range(9):
        f.suptitle('u{} EE plots with refined spokes'.format(x_axis))
        for y_axis, ax in enumerate(axes):
            x = data[x_axis, y_axis]['points'][0]
            y = data[x_axis, y_axis]['points'][1]
            ax.scatter(x, y)
            ax.set_xlabel('u{}'.format(x_axis))
            ax.set_ylabel('u{}'.format(y_axis))

            for x, y in data[x_axis, y_axis]['spoke']:
                ax.scatter(x, y)

        print('Saving figure', x_axis)
        plt.savefig('{}/u{}.png'.format(save_path, x_axis))

        for ax in axes:
            ax.cla()


def plot_u(data, u, filename):
    useful_u_indices = filter_singular_vectors(data)
    plt.imshow(u[:,0:9], interpolation='nearest', aspect='auto')
    plt.tight_layout()
    plt.title('heatmap of u. found: ' + ' '.join(map(str, useful_u_indices)))
    save_path = '{}/{}/u_visual.png'.format(PLOTS_PATH, filename)
    plt.savefig(save_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    # make all necessary dirs
    for directory in PATHS:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    filename = sys.argv[1]
    graph = read_data(filename)
    filename = filename.split('/')[-1].split('.')[0]

    u, s, v = svd(graph, filename)
    spokes = find_spokes(nx.Graph(graph), u, filename)
    data = refine_pairwise_spokes(u, spokes)
    useful_u_indices = filter_singular_vectors(data)
    plot_u(data, u, filename)
    plot_spokes(data)

