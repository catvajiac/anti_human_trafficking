#!/usr/bin/env python3

import sys

from collections import Counter, defaultdict
import numpy as np


def usage(code):
    print('Usage: _ <filename>')
    exit(code)


def read_data(filename):
    # return matrix of bigrams vs. ad id
    bigrams_to_indices = dict()
    indices_to_bigrams = dict()
    data = defaultdict(lambda: Counter())
    ads = []

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            ads.append(line)
            line = ['<s>'] + line + ['</s>']
            for a, b in zip(line, line[1:]):
                if (a, b) not in bigrams_to_indices:
                    index = len(bigrams_to_indices)
                    bigrams_to_indices[a, b] = index
                    indices_to_bigrams[index] = (a, b)

                data[i][a, b] += 1

    # construct matrix
    matrix = np.zeros((i+1, len(bigrams_to_indices)), dtype=int)
    for ad, bigrams in data.items():
        for bigram, frequency in bigrams.items():
            matrix[ad][bigrams_to_indices[bigram]] += 1
    return ads, matrix, bigrams_to_indices, indices_to_bigrams


def cluster_by_svd(matrix):
    u, s, vh = np.linalg.svd(matrix)
    seen = set()

    for i, singular_value in enumerate(s):
        # stop if all ads have already been accounted for by higher singular vals
        if len(seen) == len(s):
            return

        # find which ads are related to this singular val
        sprime = np.zeros(s.shape)
        sprime[i] = s[i]
        s_matrix = np.zeros(matrix.shape, dtype=complex)
        s_matrix[:matrix.shape[0], :matrix.shape[0]] = np.diag(sprime)

        eig_mat = np.dot(u, np.dot(s_matrix, vh))
        nonzero_entries = list(map(int, np.any(eig_mat, axis=1)))

        # filter out any ads that have already been accounted for by a different singular val
        for index, value in enumerate(nonzero_entries):
            if not value:
                continue

            if index in seen:
                nonzero_entries[index] = 0

            seen.add(index)
        yield nonzero_entries


def build_regex(matrix, ads, bigrams_to_indices, indices_to_bigrams):
    # put regexes everywhere else

    regex = [(1, token) for token in ads.pop(0)]
    regex_ngrams = construct_regex_ngrams(regex, ads)
    for row, ad in zip(matrix, ads):
        print(row, ad)
        similarity = calculate_similarity(regex_ngrams, row)
        # TODO: if similiarity bad, consider a split

        # TODO: find all common subtokens that appear in order
        derp = find_common_subtokens(regex, ad)
        print('subtok', derp)

    #print(matrix)
    pass


def find_common_subtokens(regex, ad):
    # return list of common subtokens
    # list of tuples (regex_index, ad_index, token)
    d = defaultdict(lambda: ([], []))
    print(d['hi'][0])
    for index, (ttype, token) in enumerate(regex):
        if ttype != 1:
            continue
        d[token][0].append(index)

    for index, token in enumerate(ad):
        d[token][1].append(index)


    common_indices = []
    for token, (regex_indices, ad_indices) in d.items():
        common_indices += [(r, a, token) for r, a in zip(regex_indices, ad_indices)]
    print(common_indices)
    common_indices.sort()
    prev = 0

    common_ad_indices = [x for _, x, _ in common_indices]

    marked = []
    for i, (current_i, next_i) in enumerate(zip(common_indices, common_indices[1:])):
        if current_i > next_i:
            marked = [i]

    return [x for i, x in enumerate(common_indices) if i not in marked]


def construct_regex_ngrams(regex, bigrams_to_indices):
    ngrams = np.array((1, len(bigrams_to_indices)), dtype=int)
    regex = [(1, '<s>')] + regex + [(1, '</s>')]
    for (a_type, a_token), (b_type, b_token) in zip(regex, regex[1:]):
        if (a_token, b_token) not in bigrams_to_indices:
            continue
        index = bigrams_to_indices[a_token, b_token]
        ngrams[index] += 1

    return ngrams


def calculate_similarity(regex, line):
    pass

def combine_regexes(r1, r2):
    pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    filename = sys.argv[1]
    ads, matrix, bigrams_to_indices, indices_to_bigrams = read_data(filename)
    for prelim_cluster in cluster_by_svd(matrix):
        indices = [i for i, x in enumerate(prelim_cluster) if x != 0]
        cluster_ads = [ads[i] for i in indices]
        build_regex(matrix.take(indices), cluster_ads, bigrams_to_indices, indices_to_bigrams)
