#!/usr/bin/env python3

import networkx as nx
import os, sys


def gen_graph(graph_type, params):
    graph, filename = graph_type(params)
    with open('./data/{}.txt'.format(filename), 'w') as f:
        for source, target in graph.edges:
            f.write('{} {}\n'.format(source, target))


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


def usage(code):
    print('Usage: {} [er|complete] param=num')
    exit(code)

GENERATOR = {
    'er': gen_random_graph,
    'complete': gen_complete_graph,
    'bipartite': gen_bipartite,
}

if __name__ == '__main__':
    if len(sys.argv) < 4:
        usage(1)

    if not os.path.exists('./data'):
        os.mkdir('./data')

    graph_type = GENERATOR[sys.argv[1]]

    params = {arg.split('=')[0]: arg.split('=')[1] for arg in sys.argv[2:]}
    gen_graph(graph_type, params)
