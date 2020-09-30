#!/usr/bin/env python3
# Author: Catalina Vajiac
# Purpose:
# Usage:

import os, sys
import time
import pandas

from infoshieldcoarse import *
from infoshieldfine_seq import *
from sklearn.metrics import adjusted_rand_score


def usage(code):
    print('Usage: {} [filename]'.format(os.path.basename(sys.argv[0])))
    exit(code)


def post_process_coarse(filename):
    coarse_table = pandas.read_csv('{}_full_LSH_labels.csv'.format(filename), lineterminator='\n')
    for i, small in coarse_table.groupby('LSH label'):
        if len(small) < 2000:
            coarse_table.loc[coarse_table.id.isin(small.id), 'label'] = max(small.label.values)
            #print(small.label.values, '->', coarse_table.loc[coarse_table.id.isin(small.id), 'label'].values)
        if sum(small.label.values) == 0:
            coarse_table.loc[coarse_table.id.isin(small.id), 'LSH label'] =-1

    coarse_table['is_spam'] = coarse_table.apply(lambda r: 1 if r.label >= 3  else 0, axis=1)
    #coarse_table['is_spam'] = coarse_table.apply(lambda r: 1 if r.is_spam or r.is_trafficking else 0, axis=1)
    #coarse_table['is_spam'] = coarse_table.apply(lambda r: 1 if r.is_spam or r.is_trafficking else 0, axis=1)
    field = 'is_spam'
    #coarse_table.loc[(coarse_table['LSH label'] != -1) & (coarse_table['noise'] == True), 'LSH label'] = -1
    tp = len(coarse_table[(coarse_table['LSH label'] != -1) & (coarse_table[field] == 1)])
    fp = len(coarse_table[(coarse_table['LSH label'] != -1) & (coarse_table[field] == 0)])
    fn = len(coarse_table[(coarse_table['LSH label'] == -1) & (coarse_table[field] == 1)])
    tn = len(coarse_table[(coarse_table['LSH label'] == -1) & (coarse_table[field] == 0)])

    print('tp', tp, 'fp', fp, 'fn', fn, 'tn', tn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) /(precision + recall)

    print(precision, recall, f1)


def post_process(filename):
    filename = filename.split('.')[0]
    coarse_table = pandas.read_csv('{}_full_LSH_labels.csv'.format(filename), lineterminator='\n')
    template_table = pandas.read_csv('template_table.csv')
    compression_table = pandas.read_csv('compression_rate.csv')
    #template_table = pandas.read_csv('cd_tt.csv')
    #compression_table = pandas.read_csv('cd_cr.csv')
    coarse_table['IS_is_spam'] = 0
    coarse_table['IS_cluster_label'] = -1
    if 'ht_unique' in filename or 'no_dups' in filename or 'fix' in filename:
        coarse_table['is_spam'] = coarse_table.apply(lambda r: 1 if r['label'] > 3 else 0, axis=1)
        coarse_table['real_cluster_label'] = 0
    elif 'concat' in filename:
        print('doing the right thing')
        coarse_table['is_spam'] = coarse_table.apply(lambda r: 1 if r['is_spam'] or r['is_trafficking'] else 0, axis=1)
    else:
        coarse_table['real_cluster_label'] = coarse_table.apply(lambda r: r['user_id'] if r['is_spam'] else -1, axis=1)

    total = len(coarse_table[coarse_table['LSH label'] != -1])
    tp = 0
    fp = 0
    prec_at_x = []
    recall_at_x = []
    seen = 0
    count = 0
    fn = len(coarse_table[(coarse_table.is_spam == 1) & (coarse_table.IS_is_spam == 0)])
    compression_table['Compression Rate'] = compression_table.apply(lambda r: r['Initial Cost'] / r['Final Cost'],axis=1)
    for i, row in compression_table.sort_values(by=['Compression Rate'], ascending=False).iterrows():
        if row['Compression Rate'] == 0:
            continue
        lsh_label = row['Cluster Label']
        doc_ids = template_table[template_table['LSH Label'] == lsh_label].ID
        is_spam = coarse_table[coarse_table.ad_id.isin(doc_ids.values)].is_spam
        is_t = coarse_table[coarse_table.ad_id.isin(doc_ids.values)].is_trafficking
        #print(lsh_label, doc_ids.values, is_spam)
        tp += sum(is_spam)
        fp += len(is_spam) - sum(is_spam)
        seen += len(is_spam)
        coarse_table.loc[coarse_table.ad_id.isin(doc_ids.values), 'IS_is_spam'] = 1
        prec_at_x.append(((tp+fp)/total, tp/(tp+fp)))
        recall_at_x.append(((tp+fp)/total, tp/(tp + fn)))
        #print(tp/(tp+fp), tp/(tp + fn))


    tp = len(coarse_table[(coarse_table.is_spam == 1) & (coarse_table.IS_is_spam == 1)])
    fp = len(coarse_table[(coarse_table.is_spam == 0) & (coarse_table.IS_is_spam == 1)])
    fn = len(coarse_table[(coarse_table.is_spam == 1) & (coarse_table.IS_is_spam == 0)])
    tn = len(coarse_table[(coarse_table.is_spam == 0) & (coarse_table.IS_is_spam == 0)])
    print('tp', tp, 'fp', fp, 'fn', fn, 'tn', tn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) /(precision + recall)

    for i, (_, smalldf) in enumerate(template_table.groupby(['LSH Label', 'Template #'])):
        coarse_table.loc[coarse_table.ad_id.isin(smalldf.ID), 'IS_cluster_label'] = i

    spam_table = coarse_table[coarse_table.is_spam == 1]
    sm_true_labels = spam_table['cluster_label'].values
    sm_IS_labels = spam_table['IS_cluster_label'].values
    true_labels = coarse_table['cluster_label'].values
    IS_labels = coarse_table['IS_cluster_label'].values

    print(precision, recall, f1, adjusted_rand_score(true_labels, IS_labels), adjusted_rand_score(sm_true_labels, sm_IS_labels))



if __name__ == '__main__':
    t = time.time()
    df = pandas.read_csv(sys.argv[1])
    coarse = InfoShieldCoarse(sys.argv[1])
    for g, smalldf in df.groupby('batch_num', sort=True):
        coarse.process_batch(g)
        filename = sys.argv[1].split('.')[0]
        filename += '_LSH_batch.csv' if g else '_LSH_labels.csv'
        run_infoshieldfine(filename)
