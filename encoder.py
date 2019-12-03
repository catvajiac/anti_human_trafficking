#!/usr/bin/env python3

from itertools import chain
import math
import gamma
import sys
import re


DELIMITER = ','


def usage(code):
    print("Usage: {} filename".format("encoder.py"))
    exit(code)


def pairwise(line):
    # yield pairs of (regex_id, regex) from line
    components = line.strip().split(DELIMITER)
    for type_id, string in zip(components[::2], components[1::2]):
        yield int(type_id), string


def process_file(filename):
    data = []
    regexes = []
    with open(filename, 'r') as f:
        for line in f:
            string, regex = line.split('|')
            data.append(string)
            regexes.append([pair for pair in pairwise(regex)])

    return data, regexes


def encode_regex(string, regex):
    # <id> <len> <str>
    # make bitstring to determine length
    bitstring = gamma.encoder(len(regex))
    for type_id, string in regex:
        to_compress = [type_id, len(string)] + list([ord(s) for s in string])
        bitstring += ''.join(map(gamma.encoder, to_compress))

    return bitstring


def reconstruct_string(regexes, disjunctions):
    string = ''
    for regex_id, regex in regexes:
        if regex_id == 1:
            string += regex
            continue

        string += disjunctions.pop(0)

    return string


def decoder(bitstring):
    # format to decode:
    # <# regex pieces (n)> <id_0> <len_0> <regex_0> ... <id_n-1> <len_n-1> <regex_n-1>
    # <# disjunctions (m)> <len_0> <disjunction_0> ... <len_m=1> <disjunction_m-1>
    nums = gamma.decode_stream(bitstring) # returns list of integers
    regexes, nums = decode_regex(nums)
    disjunctions = decode_disjunctions(nums)

    string = reconstruct_string(regexes, disjunctions.copy())

    return regexes, disjunctions, string


def decode_regex(nums):
    num_regexes = nums.pop(0)
    regexes = []
    for _ in range(num_regexes):
        regex_id = nums.pop(0)
        regex_len = nums.pop(0)
        regex = ''.join([chr(nums.pop(0)) for _ in range(regex_len)])
        regexes.append((regex_id, regex))

    return regexes, nums


def decode_disjunctions(nums):
    num_disjunctions = nums.pop(0)
    disjunctions = []
    for _ in range(num_disjunctions):
        disjunction_length = nums.pop(0)
        disjunction = ''.join([chr(nums.pop(0)) for _ in range(disjunction_length)])
        disjunctions.append(disjunction)

    return disjunctions


def find_disjunctions(string, regex):
    # find all substrings and regexes that correspond to each other
    # ignoring constant string regexes and their correspoinding substrings
    total_regex = '(' + ')('.join(r for _, r in regex) + ')'
    match = re.match(total_regex, string)

    disjunctions = []
    for i, (regex_id, subregex) in enumerate(regex):
        if regex_id == 1:
            continue

        substring = match.group(i+1)
        disjunctions.append(substring)

    return disjunctions


def encode_error(string, regex):
    disjunctions = find_disjunctions(string, regex)
    bitstring = gamma.encoder(len(disjunctions))
    for d in disjunctions:
        to_compress = [len(d)] + list([ord(s) for s in d])
        bitstring += ''.join(map(gamma.encoder, to_compress))

    return bitstring


def encoder(string, regex):
    return encode_regex(string, regex) + encode_error(string, regex)


def MDL(string, regex):
    return math.ceil(len(encoder(string, regex)) / 8)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage(1)

    filename = sys.argv[1]
    data, regexes = process_file(filename)

    for regex, string in zip(regexes, data):
        bits = encode_regex(string, regex)
        disjunction = encode_error(string, regex)
        #print(len(bits) + len(disjunction))

        regexes, disjunctions, string = decoder(encoder(string, regex))
        #print(regexes)
        #print(disjunctions)
        #print(string)
        print(MDL(string, regex), 'bytes')
