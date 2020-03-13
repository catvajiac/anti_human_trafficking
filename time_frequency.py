#!/usr/bin/env python3

import scipy
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
from collections import Counter
from datetime import datetime
import math
import numpy as np
import os
import pandas
import sys

# each grouping corresponds to a function returning only that relevant part of the datetime object
GROUP_BY = {
    'day': lambda x: x.strftime("%Y %m %d"),
    'month': lambda x: x.strftime("%Y %m"),
    'year': lambda x: x.year
}

TICKS = {
    'day': (10, 30),
    'month': (1, 3),
    'year': (1, 1)
}


POPULATIONS = {
    'barrie': 153356,
    'vernon': 40116,
    'skeena': 90586,
    'cariboo': 61988,
    'medicinehat': 63260,
    'owensound': 21341,
    'whistler': 11854,
    'fredericton': 58220,
    'princegeorge': 74003,
    'cranbrook': 19259,
    'brandon': 48859,
    'ftmcmurray': 66573,
    'stalbert': 65589,
    'yellowknife': 19569,
    'sarnia': 71594,
    'thunderbay': 110172,
    'comoxvalley': 66527,
    'reddeer': 103588,
    'grandeprairie': 63166,
    'lethbridge': 92730,
    'nanaimo': 90505,
    'abbotsford': 149928,
    'moncton': 85198,
    'oshawa': 170071,
    'sudbury': 164926,
    'stjohns': 113948,
    'peterborough': 84230,
    'kamloops': 90280,
    'kelowna': 132084,
    'kingston': 136685,
    'victoria': 92141,
    'london': 404699,
    'halifax': 431479,
    'northbay': 51553,
    'vancouver': 675218,
    'winnipeg': 749534,
    'kitchener': 242368,
    'ottawa': 994837,
    'niagara': 447888,
    'edmonton': 981280,
    'calgary': 1336000,
    'hamilton': 579200,
    'toronto': 2930000
        }


def read_canadian_data(filename):
    data = pandas.read_csv(filename)
    return data


def read_mit_data():
    # send query
    pass


def plot_time_frequency(filename, data, frequency='month'):
    # strip directory and filetype
    filename_base = filename.split('/')[-1].split('.')[0]
    strip_date = lambda x: datetime.strptime(x, '%m-%d-%Y')

    # apply grouping function to each date in dataset, group by frequency
    group_count = Counter([GROUP_BY[frequency](strip_date(x)) for x in data['date']])
    group_count = sorted(group_count.items())

    # plot: set x axis so labels don't overlap
    min_ticks, max_ticks = TICKS[frequency]
    plt.figure(figsize=(20, 10))
    ax = plt.axes()
    ticker.Locator.MAXTICKS = 2100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(max_ticks))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(min_ticks))

    plt.ylabel('frequency')
    plt.xlabel('time ({}s)'.format(frequency))
    plt.title('Time-frequency plot: grouped by {}s'.format(frequency))

    x = [t[0] for t in group_count]
    y = [t[1] for t in group_count]

    prev_x, prev_y = -1, -1
    for x_, y_ in zip(x, y):
        if abs(prev_y - y_) > 30:
            print('Prev: ', prev_x, prev_y)
            print('Curr: ', x_, y_)
            print()
        prev_x = x_
        prev_y = y_

    plt.plot(x, y)
    plt.xticks(rotation=70, fontsize=10)
    plt.savefig('./plots/' + filename_base + '_' + frequency)
    plt.show()


def plot_deltas(data):
    # apply grouping function to each date in dataset, group by frequency
    strip_date = lambda x: datetime.strptime(x, '%m-%d-%Y')
    filename_base = filename.split('/')[-1].split('.')[0]
    group_count = Counter([GROUP_BY['day'](strip_date(x)) for x in data['date']])
    group_count = sorted(group_count.items())

    plt.scatter([y for _, y in group_count[:-1]], [y for _, y in group_count[1:]])
    plt.title('freq at t vs. freq at t+1')
    plt.xticks(rotation=70, fontsize=10)
    plt.savefig('./plots/' + filename_base + '_' + 'deltas' + 'day')
    plt.show()


def fft(data, frequency='month'):
    strip_date = lambda x: datetime.strptime(x, '%m-%d-%Y')

    # apply grouping function to each date in dataset, group by frequency
    group_count = Counter([GROUP_BY[frequency](strip_date(x)) for x in data['date']])
    time_data = [v for _, v in group_count.items()]
    x = scipy.fft(time_data)
    print(x)


def print_phone_data(data):
    phone = set(data['phone'])
    for p in phone:
        if type(p) == str and p.startswith('000'):
            print(p)

    d = data.groupby('phone')
    for thing in d:
        print(thing)


def create_time_plots(data, filename):
    for col in data.columns.values:
        c = Counter(data[col])
        c = sorted(c.items(), key=lambda x: x[1])
        for tup in c[-10:]:
            print(tup)
        print()


    for frequency in GROUP_BY:
        plot_time_frequency(filename, data, frequency)
    plot_deltas(data)


def create_location_plots(data, filename):
    #for col in ('location', 'default_location'):
    c = Counter(data['default_location'])
    c_sort = sorted(c.items(), key=lambda x: x[1])
    for tup in c_sort[-10:]:
        print(tup)
    print()

    # HERE
    cities = [math.log10(tup[1]) for tup in c_sort]
    nums = [math.log10(POPULATIONS[tup[0]]) for tup in c_sort]
    names = [tup[0] for tup in c_sort]
    print(cities)
    print(nums)


    plt.rcParams.update({'font.size': 7})
    plt.figure(figsize=(20, 10))
    plt.scatter(nums, cities)
    plt.xticks(rotation=70, fontsize=10)
    plt.title('# ads in city vs. population size')
    plt.xlabel('Population Size')
    plt.ylabel('Number of Ads')
    for name, pop, city in zip(names, nums, cities):
        plt.annotate(name, (pop, city), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.show()


def usage(exit_code):
    print('Usage: _ [mode] [filename]')
    print('-m    use MITLL data')
    print('-c    use Canadian data (must follow with filename)')
    exit(exit_code)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)

    mode = sys.argv[1]

    if mode not in ('-m', '-c') or mode == '-c' and len(sys.argv) < 3:
        usage(1)

    if not os.path.isdir('./plots'):
        os.mkdir('./plots')


    if mode == '-c':
        filename = sys.argv[2]
        data = read_canadian_data(filename)
    if mode == '-m':
        filename = 'MITLL'
        data = read_mit_data()

    create_location_plots(data, filename)
