#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os, sys
import pickle

from collections import Counter, defaultdict
from sortedcontainers import SortedList


# global path vars

PLOTS_PATH = './plots/eigenspokes/'
PKL_PATH = './pkl_files'
PATHS = [PLOTS_PATH, PKL_PATH]


def read_data(filename):
    ''' Assume data format is edge list split by newlines, autoconverts labels to integers '''
    with open(filename, 'r') as f:
        edges = [tuple(line.strip().split()) for line in f]
    graph = nx.Graph(edges)
    #graph = nx.convert_node_labels_to_integers(graph)
    return graph


def svd(graph, filename):
    ''' returns tuple: (u, s, v) '''
    print('Running SVD...')
    svd_pkl_filename = '{}/{}_svd.pkl'.format(PKL_PATH, filename)
    if os.path.exists(svd_pkl_filename):
        return pickle.load(open(svd_pkl_filename, 'rb'))

    array = nx.to_numpy_matrix(graph)
    u, s, v = np.linalg.svd(array)
    u = np.array(u)
    pickle.dump((u, s, v), open(svd_pkl_filename, 'wb'))
    return u, s, v


def modularity(graph):
    ''' returns number of communities found '''
    return len(nx.algorithms.community.greedy_modularity_communities(graph))


def find_spoke(scores, graph, metric=modularity):
    ''' first: find the point with highest projection, that is current spoke/subgraph
        then: always process a neighboring node to current subgraph which has the highest proj
        stop: when modularity metric says there are multiple communities '''
    nodes_to_scores = {i: x for i, x in scores}
    visited = set()

    key = lambda tup: tup[1]
    i, x = min(scores, key=key)
    neighbors = SortedList([(n, nodes_to_scores[n]) for n in  graph.neighbors(i)], key=key)
    spoke = set([i])

    while len(neighbors):
        i, x = neighbors.pop(-1)
        spoke.add(i)
        visited.add(i)

        spoke_graph = graph.subgraph(spoke)
        if metric(spoke_graph) > 1:
            spoke.remove(i)
            continue

        for neighbor in graph.neighbors(i):
            if neighbor in visited:
                continue
            neighbors.add((neighbor, nodes_to_scores[neighbor]))

    return spoke


def find_spokes(graph, u, filename):
    ''' Find spokes - for now just looks at top 9 eigenvectors for ease of plotting later '''
    print('Finding spokes...')
    spokes_pkl_filename = '{}/{}_spokes.pkl'.format(PKL_PATH, filename)
    if os.path.exists(spokes_pkl_filename):
        return pickle.load(open(spokes_pkl_filename, 'rb'))

    spokes = []
    for axis in range(9):
        scores = [(i, x) for i, x in enumerate(u[:, axis])]
        spokes.append([s for s in find_spoke(scores, graph)])
        print('  u{} spoke found. size: {}'.format(axis, len(spokes)))

    pickle.dump(spokes, open(spokes_pkl_filename, 'wb'))
    return spokes


def plot_spokes(u, spokes):
    ''' plots EE plots of the first 9 eigenvectors '''
    array = nx.to_numpy_matrix(graph)
    u, s, v = np.linalg.svd(array)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3,  sharex='col', sharey='row', figsize=(8,8), dpi=80)

    u = np.array(u)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    save_path = '{}/{}'.format(PLOTS_PATH, filename)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for x_axis in range(9):
        u_x = u[:, x_axis]
        f.suptitle('{} comparisons with u{}'.format(filename, x_axis))

        for y_axis, ax in enumerate(axes):
            u_y = u[:, y_axis]
            ax.scatter(u_x, u_y)
            ax.set_xlabel('u{}'.format(x_axis))
            ax.set_ylabel('u{}'.format(y_axis))
            print('({}, {})'.format(x_axis, y_axis))

            '''
            for direction in [x_axis, y_axis]:
                spoke = spokes[direction]
                x = u[spoke, x_axis]
                y = u[spoke, y_axis]
                ax.scatter(x, y)
                print('plotted a spoke of size ', len(x))
            '''


        print('Saving figure', x_axis)
        plt.savefig('{}/u{}.png'.format(save_path, x_axis))

        for ax in axes:
            ax.cla()


def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


def example(G):
    A = nx.to_numpy_matrix(G)

    u, s, v = np.linalg.svd(A)
    u = np.array(u)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3,  sharex='col', sharey='row', figsize=(8,8), dpi=80)

    ax1.scatter(u[:,0],u[:,0])
    ax2.scatter(u[:,0],u[:,1])
    ax3.scatter(u[:,0],u[:,2])
    ax4.scatter(u[:,0],u[:,3])
    ax5.scatter(u[:,0],u[:,4])
    ax6.scatter(u[:,0],u[:,5])
    ax7.scatter(u[:,0],u[:,6])
    ax8.scatter(u[:,0],u[:,7])
    ax9.scatter(u[:,0],u[:,8])

    plt.show()


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

    #u, s, v = svd(graph, filename)
    #spokes = find_spokes(graph, u, filename)
    #plot_spokes(None, [])
    example(graph)
