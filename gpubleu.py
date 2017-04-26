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
    
    def _convert_counts(self, seqs, cmap):
        sizes = map(len, cmap)
        offset_2 = sizes[0]
        offset_3 = sum(sizes[:2])
        offset_4 = sum(sizes[:3])
        matrix = np.zeros((len(seqs), sum(sizes)), dtype=FLOATX)
        for n, seq in enumerate(seqs):
            for i in range(len(seq)):
                if seq[i] in cmap[0]:
                    id = cmap[0][seq[i]]
                    matrix[n, id] += 1.
                if i >= 1:
                    t = tuple(seq[i - 1:i + 1])
                    if t in cmap[1]:
                        id = cmap[1][t]
                        matrix[n, offset_2 + id] += 1.
                if i >= 2:
                    t = tuple(seq[i - 2:i + 1])
                    if t in cmap[2]:
                        id = cmap[2][t]
                        matrix[n, offset_3 + id] += 1.
                if i >= 3:
                    t = tuple(seq[i - 3:i + 1])
                    if t in cmap[3]:
                        id = cmap[3][t]
                        matrix[n, offset_4 + id] += 1.
        return matrix
    
    def precompute_ngram_map(self, seqs):
        cmap = [{}, {}, {}, {}]
        for seq in seqs:
            for i in range(len(seq)):
                if seq[i] not in cmap[0]:
                    cmap[0][seq[i]] = len(cmap[0])
                if i >= 1:
                    t = tuple(seq[i - 1:i + 1])
                    if t not in cmap[1]:
                        cmap[1][t] = len(cmap[1])
                if i >= 2:
                    t = tuple(seq[i - 2:i + 1])
                    if t not in cmap[2]:
                        cmap[2][t] = len(cmap[2])
                if i >= 3:
                    t = tuple(seq[i - 3:i + 1])
                    if t not in cmap[3]:
                        cmap[3][t] = len(cmap[3])
        return cmap
        
    def evaluate(self, hyps, refs, ngram_map=None):
        if ngram_map is None:
            if len(hyps) < len(refs):
                ngram_map = self.precompute_ngram_map(hyps)
            else:
                ngram_map = self.precompute_ngram_map(refs)
        seps = map(len, ngram_map)
        tok_lens = np.array(map(len, hyps), dtype=FLOATX)
        ev_lens = np.array(map(len, refs), dtype=FLOATX)

        matrix = self._convert_counts(hyps, ngram_map)
        ev_matrix = self._convert_counts(refs, ngram_map)
        
        
        bleu_matrix = self._graph(*([matrix, ev_matrix] + seps + [tok_lens, ev_lens]))
        return bleu_matrix


if __name__ == '__main__':
    import sys
    # Test code
    bleu = GPUBLEUEvaluator()
    sents = [[1, 2, 3, 4, 5, 3, 4, 1], [2, 2, 2, 5, 7], [1, 2, 3, 3, 4, 5, 1, 5, 6, 7, 4, 3, 3, 6], [1, 2, 3, 4, 5], [1, 2, 3, 7, 7, 7]] * 1000
    ref = [[1, 2, 3, 4, 5, 7]]
    K = 1000
    
    print ("evaluate {} hyps with {} refs for {} times".format(len(sents), len(ref), K))
    
    cmap = bleu.precompute_ngram_map(ref)
    for _ in range(K):
         bleu.evaluate(sents, ref, ngram_map=cmap)
         sys.stdout.write(".")
         sys.stdout.flush()
