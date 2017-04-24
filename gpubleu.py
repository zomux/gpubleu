#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

FLOATX = 'float32'

class GPUBLEUEvaluator(object):
    
    def __init__(self):
        self.gram_weights = theano.shared(np.array([0.25, 0.25, 0.25, 0.25], dtype=FLOATX))
        g, r, m1, m2, m3, m4, nx, nr = [getattr(T, n)() for n in ['matrix', 'matrix', 'iscalar', 'iscalar', 'iscalar', 'iscalar', 'vector', 'vector']]
        
        j = T.lt(g[:, None, :] - r[None, :, :], 0)
        mat = j * g[:, None, :] + (1 - j) * r[None, :, :]
        dn1 = (nx[:, None] + 1 - 1)
        log1 = T.log((1 + mat[:, :, 0: m1].sum(axis=2)) / (T.gt(dn1, 0) * dn1 + 1))
        dn2 = (nx[:, None] + 1 - 2)
        log2 = T.log((1 + mat[:, :, m1: m1 + m2].sum(axis=2)) / (T.gt(dn2, 0) * dn2 + 1))
        dn3 = (nx[:, None] + 1 - 3)
        log3 = T.log((1 + mat[:, :, m1 + m2: m1 + m2 + m3].sum(axis=2)) / (T.gt(dn3, 0) * dn3 + 1))
        dn4 = (nx[:, None] + 1 - 4)
        log4 = T.log((1 + mat[:, :, m1 + m2 + m3: m1 + m2 + m3 + m4].sum(axis=2)) / (T.gt(dn4, 0) * dn4 + 1))
        log_vals = [log1, log2, log3, log4]
        log_sum = sum([log_vals[i] * self.gram_weights[i] for i in range(4)])
        first_part = 1 - nr[None, :] / nx[:, None]
        bleu_matrix = T.exp(T.lt(first_part, 0) * first_part + log_sum)
        
        # Raw graph
        self._graph = theano.function([g, r, m1, m2, m3, m4, nx, nr], bleu_matrix)
    
    def _get_counts(self, seqs, cmap=None):
        if cmap is None:
            cmap = [{}, {}, {}, {}]
        cnt_list = []
        for seq in seqs:
            cnt = [{}, {}, {}, {}]
            for i in range(len(seq)):
                if seq[i] not in cmap[0]:
                    cmap[0][seq[i]] = len(cmap[0])
                id = cmap[0][seq[i]]
                if id not in cnt[0]:
                    cnt[0][id] = 0
                cnt[0][id] += 1
                if i >= 1:
                    t = tuple(seq[i - 1:i + 1])
                    if t not in cmap[1]:
                        cmap[1][t] = len(cmap[1])
                    id = cmap[1][t]
                    if id not in cnt[1]:
                        cnt[1][id] = 0
                    cnt[1][id] += 1
                if i >= 2:
                    t = tuple(seq[i - 2:i + 1])
                    if t not in cmap[2]:
                        cmap[2][t] = len(cmap[2])
                    id = cmap[2][t]
                    if id not in cnt[2]:
                        cnt[2][id] = 0
                    cnt[2][id] += 1
                if i >= 3:
                    t = tuple(seq[i - 3:i + 1])
                    if t not in cmap[3]:
                        cmap[3][t] = len(cmap[3])
                    id = cmap[3][t]
                    if id not in cnt[3]:
                        cnt[3][id] = 0
                    cnt[3][id] += 1
            cnt_list.append(cnt)
        return cnt_list, cmap
    
    def _convert_counts(self, cnt_list, cmap):
        cnt_vecs = []
        sizes = map(len, cmap)
        sum_size = sum(sizes)
        for cnt in cnt_list:
            vec = np.zeros(sum_size, dtype="int32")
            base_idx = 0
            for i in range(4):
                for id, c in cnt[i].items():
                    vec[id + base_idx] = c
                base_idx += sizes[i]
            cnt_vecs.append(vec)
        matrix = np.stack(cnt_vecs)
        return matrix.astype(FLOATX), sizes
    
    def evaluate(self, hyps, refs):
        tok_lens = np.array(map(len, hyps), dtype=FLOATX)
        ev_lens = np.array(map(len, refs), dtype=FLOATX)
        ev_cnt_list, cmap = self._get_counts(refs)
        cnt_list, cmap = self._get_counts(hyps, cmap=cmap)
        ev_matrix, seps = self._convert_counts(ev_cnt_list, cmap)
        matrix, seps = self._convert_counts(cnt_list, cmap)
        bleu_matrix = self._graph(*([matrix, ev_matrix] + seps + [tok_lens, ev_lens]))
        return bleu_matrix


if __name__ == '__main__':
    # Test code
    bleu = GPUBLEUEvaluator()
    sents = [[1, 2, 3, 4, 5, 3, 4, 1], [2, 2, 2, 5, 7], [1, 2, 3, 3, 4, 5, 1, 5, 6, 7, 4, 3, 3, 6]]
    ref = [[1, 2, 3, 4, 5, 7]]
    print (bleu.evaluate(sents, ref))
