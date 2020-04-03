#!/usr/bin/env python3
# Author: Catalina
# Purpose: generate graph data for experiments for AHT
# Usage: ./graph_generator.py [graph_type] [necessary params]

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import networkx as nx
import os, sys
import pickle


def gen_graph(graph_type, params):
    graph, filename = graph_type(params)
    with open('./data/{}.txt'.format(filename), 'w') as f:
        for source, target in graph.edges:
            f.write('{} {}\n'.format(source, target))

    pickle.dump(graph, open('./data/{}.pkl'.format(filename), 'wb'))



def read_data(filename):
    with open(filename, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
        graph = nx.DiGraph(edges)
        return graph


def gen_complete_graph(params):
    return nx.complete_graph(params['nodes']), 'complete-{}'.format(nodes)


def gen_random_graph(params):
    nodes = int(params['nodes'])
    edges = int(params['edges'])
    return nx.gnm_random_graph(nodes, edges), 'er_nodes-{}_edges-{}'.format(nodes, edges)


def gen_bipartite(params):
    n = int(params['n'])
    m = int(params['m'])
    return nx.complete_multipartite_graph(n, m), 'bipartite_n-{}_m-{}'.format(n, m)


def gen_one_block(x, y, density):
    size = max(x, y)
    matrix = sp.sparse.random(x, y, density=density)
    matrix.data[:] = 1
    return matrix.todense()


def gen_blocks(params):
    #block_sizes = [(10, 20, 0.75), (20, 30, 0.7), (20, 10, 0.8)]
    block_sizes = [(50, 100, 0.75), (100, 150, 0.7), (100, 50, 0.8)]

    is_noise = params['noise'] == 'True' if 'noise' in params else False
    is_camouflage = params['camouflage'] == 'True' if 'camouflage' in params else False

    if is_noise:
        block_sizes.append((150, 100, 0.1))
    max_nodes = max(sum(x[0] for x in block_sizes), sum(x[1] for x in block_sizes))

    matrix = np.zeros((max_nodes, max_nodes))
    coordx, coordy = (0, 0)
    for x, y, density in block_sizes:
        temp_matrix = gen_one_block(x, y, density)
        matrix[coordx:coordx+x, coordy:coordy+y] = temp_matrix
        coordx += x
        coordy += y

    graph = nx.DiGraph(matrix)

    if is_camouflage:
        camouflage = nx.fast_gnp_random_graph(max_nodes, 0.005, directed=True)
        graph.add_edges_from(camouflage.edges)

    graph = nx.convert_node_labels_to_integers(graph)
    matrix = nx.to_numpy_matrix(graph)
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.title('Heatmap of input data matrix')
    filename = 'dense_noise-{}_camouflage-{}'.format(is_noise, is_camouflage)
    plt.savefig('./plots/eigenspokes/' + filename + '/heatmap.png')
    return graph, filename


def usage(code):
    print('Usage: {} [er|complete|bipartite|dense] param=num')
    exit(code)

GENERATOR = {
    'er': gen_random_graph,
    'complete': gen_complete_graph,
    'bipartite': gen_bipartite,
    'dense': gen_blocks
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    if not os.path.exists('./data'):
        os.mkdir('./data')

    graph_type = GENERATOR[sys.argv[1]]

    params = {arg.split('=')[0]: arg.split('=')[1] for arg in sys.argv[2:]}
    gen_graph(graph_type, params)
