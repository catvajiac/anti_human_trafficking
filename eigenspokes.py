#!/usr/bin/env python3
# Author:   Catalina Vajiac
# Purpose:  decides which eigenvectors store ''useful'' information
# Usage:    ./eigenspokes.py [filename]

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os, sys
import pickle
import warnings

from collections import defaultdict
from itertools import product
from sortedcontainers import SortedList
from scipy.stats import multivariate_normal
import scipy as sp


# Utility functions

def usage(exit_code):
    print('Usage: _ [filename]')
    exit(exit_code)


def modularity(subgraph, graph):
    ''' returns number of communities found '''
    return len(nx.algorithms.community.greedy_modularity_communities(graph)) > 1

def conductance(subgraph, graph):
    return nx.algorithms.cuts.conductance(graph, subgraph.nodes) < 0.93


def filter_zeros(u_x, u_y, zero_cutoff):
    ''' removes pairs of x and y that are close to zero '''
    nonzero = lambda x, y: abs(x) > zero_cutoff and abs(y) > zero_cutoff
    plot_x = [x for x, y in zip(u_x, u_y) if nonzero(x, y)]
    plot_y = [y for x, y in zip(u_x, u_y) if nonzero(x, y)]

    return plot_x, plot_y


def calc_entropy(x, y):
    ''' fits a 2D gaussian to point cloud and calculates the entropy of the data given the
        gaussian model '''
    # note: have to add noise to make cov matrix not singular
    data = np.stack((x, y), axis=0)# + .001*np.random.rand(2, len(x))
    # sometimes low degree of freedom warning, ok in our case
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cov = np.cov(data)

    try:
        mean = [sum(x) / len(x), sum(y) / len(y)]
        entropy = abs(multivariate_normal(mean=mean, cov=cov).entropy())
    except:
        entropy = float('inf')

    return entropy


def flip_svd_order(u, s, v):
    n = len(s)
    u[:,:n] = u[:, n-1::-1]
    s = s[::-1]
    v[:n, :] = v[n-1::-1, :]
    return u, s, v


# main class

class find_spokes():
    def __init__(self, filename, pkl_path='./pkl_files', plots_path='./plots/eigenspokes', zero_cutoff=1e-2):
        self.read_data(filename)
        self.filename = os.path.basename(filename).split('.')[0]
        self.pkl_path = './pkl_files'
        self.plots_path = '{}/{}'.format(plots_path, self.filename)
        self.zero_cutoff = 1e-2


    def read_data(self, filename):
        ''' reads networkx pkl or edgelist split by newlines, autoconverts labels to integers '''
        if filename.endswith('.pkl'):
            self.graph = pickle.load(open(filename, 'rb'))
            return

        with open(filename, 'r') as f:
            edges = [tuple(map(int, line.strip().split())) for line in f]
        graph = nx.DiGraph(edges)
        self.graph = nx.convert_node_labels_to_integers(graph)


    def svd(self, use_pkl=True):
        ''' returns tuple: (u, s, v) '''
        svd_pkl_filename = '{}/{}_svd.pkl'.format(self.pkl_path, self.filename)
        if use_pkl and os.path.exists(svd_pkl_filename):
            print('Using SVD pkl file...')
            self.u, self.s, self.v = pickle.load(open(svd_pkl_filename, 'rb'))
            return

        print('Running SVD...')
        array = nx.to_scipy_sparse_matrix(self.graph, dtype='float64')
        self.u, self.s, self.v = flip_svd_order(*sp.sparse.linalg.svds(array, k=9, which='LM'))
        if use_pkl:
            pickle.dump((self.u, self.s, self.v), open(svd_pkl_filename, 'wb'))


    def find_spoke(self, scores, metric=conductance):
        ''' first: find the point with highest projection, that is current spoke/subgraph
            then: always process a neighboring node to current subgraph which has the highest proj
            stop: when modularity metric says there are multiple communities '''
        nodes_to_scores = {i: x for i, x in scores}
        visited = set()

        key = lambda tup: tup[1]
        i, x = max(scores, key=key)
        neighbors = SortedList([(n, nodes_to_scores[n]) for n in  self.graph.neighbors(i)], key=key)
        spoke = set([i])

        while len(neighbors):
            i, x = neighbors.pop(-1)
            visited.add(i)
            if x <= self.zero_cutoff:
                continue
            spoke.add(i)

            spoke_graph = self.graph.subgraph(spoke)
            if conductance(spoke_graph, self.graph):
                spoke.remove(i)
                break

            for neighbor in self.graph.neighbors(i):
                score = nodes_to_scores[neighbor]
                # neighbor could be in neighbors: i.e. waiting to be processed
                if neighbor in visited or (neighbor, score) in neighbors:
                    continue
                neighbors.add((neighbor, nodes_to_scores[neighbor]))

        return list(spoke)


    def find_spokes(self, use_pkl=True):
        ''' Find spokes - for now just looks at top 9 eigenvectors for ease of plotting later '''
        spokes_pkl_filename = '{}/{}_spokes.pkl'.format(self.pkl_path, self.filename)
        if use_pkl and os.path.exists(spokes_pkl_filename):
            print('Using spokes pkl file...')
            self.spokes = pickle.load(open(spokes_pkl_filename, 'rb'))
            return

        print('Finding spokes...')
        self.spokes = []
        for axis in range(9):
            if self.u.shape[0] <= axis:
                break
            scores = [(i, abs(x)) for i, x in enumerate(self.u[:, axis])]
            self.spokes.append(self.find_spoke(scores, self.graph))
            print('  u{} spoke found. size: {}'.format(axis, len(self.spokes[-1])))

        if use_pkl:
            pickle.dump(self.spokes, open(spokes_pkl_filename, 'wb'))


    def refine_pairwise_spokes(self):
        ''' only keeps spokes that create useful EE plots '''
        data = {(x, y): defaultdict(list) for x, y in product(range(9), repeat=2)}

        for x_axis in range(9):
            for y_axis in range(9):
                if self.u.shape[0] <= x_axis or self.u.shape[0] <= y_axis:
                    data.pop((x_axis, y_axis))
                    continue
                x, y = filter_zeros(self.u[:, x_axis], self.u[:, y_axis], self.zero_cutoff)
                data[x_axis, y_axis]['points'] = (x, y)

                for direction in [x_axis, y_axis]:
                    spoke = self.spokes[direction]
                    x = self.u[spoke, x_axis]
                    y = self.u[spoke, y_axis]
                    if not len(x) or calc_entropy(x, y) < 5:
                        continue
                    data[x_axis, y_axis]['spoke'].append((x, y))

        self.data = data


    def filter_singular_vectors(self):
        ''' this returns a set of singular vectors that create spokes in EE plots with all other
            singular vectors in the set '''
        max_sv = max([max(x, y) for x, y in self.data]) + 1
        singular_vectors = set()
        for x in range(9):
            if sum(([any(self.data[x, y]['spoke']) for y in range(max_sv)])) == 9:
                singular_vectors |= set([x])
                continue
            spokes = set([y for y in range(max_sv) if y != x and len(self.data[x, y]['spoke'])])
            singular_vectors |= spokes

        self.filtered_vectors = singular_vectors


    def plot_spokes(self):
        ''' plots EE plots of the first 9 eigenvectors '''

        f, ((a1, a2, a3), (a4, a5, a6), (a7, a8, a9))= plt.subplots(3, 3,  sharex='col', sharey='row', figsize=(8,8), dpi=80)

        axes = [a1, a2, a3, a4, a5, a6, a7, a8, a9]

        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

        for x_axis in range(9):
            f.suptitle('u{} EE plots with refined spokes'.format(x_axis))
            for y_axis, ax in enumerate(axes):
                x = self.data[x_axis, y_axis]['points'][0]
                y = self.data[x_axis, y_axis]['points'][1]
                ax.scatter(x, y)
                ax.set_xlabel('u{}'.format(x_axis))
                ax.set_ylabel('u{}'.format(y_axis))

                for x, y in self.data[x_axis, y_axis]['spoke']:
                    ax.scatter(x, y)

            print('Saving figure', x_axis)
            plt.savefig('{}/u{}.png'.format(self.plots_path, x_axis))

            for ax in axes:
                ax.cla()


    def plot_u(self):
        self.filter_singular_vectors()
        plt.imshow(self.u[:,0:9], interpolation='nearest', aspect='auto')
        plt.tight_layout()
        plt.title('heatmap of u. found: ' + ' '.join(map(str, self.filtered_vectors)))
        save_path = '{}/u_visual.png'.format(self.plots_path)
        plt.savefig(save_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    spokes = find_spokes(sys.argv[1])
    spokes.svd()
    spokes.find_spokes()
    spokes.refine_pairwise_spokes()
    spokes.filter_singular_vectors()
    spokes.plot_u()
    spokes.plot_spokes()
