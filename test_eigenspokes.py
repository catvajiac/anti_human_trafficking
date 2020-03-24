#!/usr/bin/env python3

import eigenspokes
import os, sys
import unittest

from itertools import product


class test_eigenspokes(unittest.TestCase):
    def test_read_data(self, filename='test_eigenspokes.txt'):
        graph = eigenspokes.read_data(filename)
        self.assertEqual(graph.number_of_nodes(), 8)
        self.assertEqual(graph.number_of_edges(), 24)

    def test_modularity(self, filename='test_eigenspokes.txt'):
        graph = eigenspokes.read_data(filename)
        num_communities = eigenspokes.modularity(graph)
        self.assertEqual(num_communities, 2)

    def test_filter_zeros(self):
        data = [-0.1, -0.01, -0.001, -0.0001, 0, 0.0001, 0.001, 0.01, 0.1]
        x = [tup[0] for tup in product(data, repeat=2)]
        y = [tup[1] for tup in product(data, repeat=2)]
        new_x, new_y = eigenspokes.filter_zeros(x, y)
        zero = eigenspokes.ZERO_CUTOFF
        for x, y in zip(new_x, new_y):
            self.assertTrue(abs(x) >= zero or abs(y) >= zero)

    def test_calc_entropy(self):
        data = [0, 0.0001, 0.001, 0.01, 0.1]
        x = [tup[0] for tup in product(data, repeat=2)]
        y = [tup[1] for tup in product(data, repeat=2)]
        entropy = eigenspokes.calc_entropy(x, y)
        self.assertAlmostEqual(entropy, 3.60609, places=4)

    def test_svd(self, filename='test_eigenspokes.txt'):
        graph = eigenspokes.read_data(filename)
        u, s, v = eigenspokes.svd(graph, filename, use_pkl=False)
        self.assertEqual(u.shape, v.shape, s.shape)
        for col in range(u.shape[1]):
            self.assertAlmostEqual(sum(u[:, col]**2), 1)

    def test_find_spokes(self, filename='test_eigenspokes.txt'):
        # this will subsequently also test find_spoke subroutine
        graph = eigenspokes.read_data(filename)
        u, s, v = eigenspokes.svd(graph, filename, use_pkl=False)
        spokes = eigenspokes.find_spokes(graph, u, 'test', use_pkl=False)
        for spoke in spokes:
            self.assertEqual(len(spoke), 1)

    def test_refine_pairwise_spokes(self, filename='test_eigenspokes.txt'):
        graph = eigenspokes.read_data(filename)
        u, s, v = eigenspokes.svd(graph, filename, use_pkl=False)
        spokes = eigenspokes.find_spokes(graph, u, 'test', use_pkl=False)
        data = eigenspokes.refine_pairwise_spokes(u, spokes)
        zero = eigenspokes.ZERO_CUTOFF
        for x_axis, y_axis in data:
            x_points, y_points = data[x_axis, y_axis]['points']
            for x, y in zip(x_points, y_points):
                self.assertTrue(abs(x) >= zero or abs(y) >= zero)
            for x, y in data[x_axis, y_axis]['spoke']:
                self.assertTrue(eigenspokes.calc_entropy(x, y) >= 5)


    def test_filter_singular_vectors(self, filename='test_eigenspokes.txt'):
        graph = eigenspokes.read_data(filename)
        u, s, v = eigenspokes.svd(graph, filename, use_pkl=False)
        spokes = eigenspokes.find_spokes(graph, u, 'test', use_pkl=False)
        data = eigenspokes.refine_pairwise_spokes(u, spokes)
        filtered_spokes = eigenspokes.filter_singular_vectors(data)
        self.assertEqual(filtered_spokes, set(range(7)))


if __name__ == '__main__':
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        unittest.main(buffer=False)
