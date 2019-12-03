#!/usr/bin/env python3


def to_binary(x):
    return bin(x)[2:]


def encoder(num):
    offset = to_binary(num)[1:] # remove leading 1
    return '0'*len(offset) + '1' + offset


def find_delimiter(bitstring):
    index = 0
    while bitstring[index] == '0':
        index += 1

    return bitstring[0:index+1], bitstring[index+1:index*2+1]


def decode_stream(bitstring):
    nums = []
    while len(bitstring):
        unary, offset = find_delimiter(bitstring)
        nums.append(int('0b1' + offset, 2))
        new_index = len(unary) + len(offset)
        bitstring = bitstring[new_index:]

    return nums


if __name__ == '__main__':
    lst = [0, 17, 72, 101, 108, 108, 111, 32, 109, 121, 32, 110, 97, 109, 101, 32, 105, 115, 32]
    bitstring = ''.join(map(encoder, lst))
    #bitstring = ''.join(map(encoder, [22, 1, 2, 3, 4, 5, 17]))
    bitstring = '010100001000100000010010000000001100101000000110110000000011011000000001101111000001000000000001101101000000111100100000100000000000110111000000011000010000001101101000000110010100000100000000000110100100000011100110000010000010001010000000101101100000010000010000010110100000010110100000001100001000001011010000001111010000000101110100000101110000001010101000100000000010000110000001100001000000111010000000011000010000001101100000000110100100000011011100000001100001'
    print(decode_stream(bitstring))
