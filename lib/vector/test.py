# Gluon starter
import mxnet as mx
import gluonnlp as nlp
from collections.abc import Mapping

class EmbeddingSample(Mapping):
    """ View of a subset of the terms in an embedding,
        such as a sampling of terms"""

    def __init__(self, embed, term_idxs):
        self.embed=embed
        self.term_idxs=term_idxs

    def __getitem__(self, token):
        idx = self.embed.glove.token_to_idx[token]
        if idx not in self.term_idxs:
            raise KeyError("{} not present in embedding".format(token))
        else:
            self.glove[token]

    def __iter__(self):
        for idx in self.term_idxs:
            yield self.embed.glove.idx_to_token[idx], self.embed.glove.idx_to_vec[idx]

    def __len__(self):
        len(self.term_idxs)


class GloveEmbeddings(Mapping):

    def __init__(self):
        self.glove = nlp.embedding.create('glove', source='glove.6B.50d')

    def __getitem__(self, token):
        """ Get an embedding, given token"""
        idx = self.glove.token_to_idx[token]
        return self.glove.idx_to_vec[idx]

    def __iter__(self):
        """ Iterate every token,embedding"""
        for idx in range(0, len(self)):
            yield self.glove.idx_to_token[idx], self.glove.idx_to_vec[idx]

    def __len__(self):
        """ Number of terms """
        return len(self.glove.idx_to_token)

    def sample(self, n=100):
        from random import sample
        return EmbeddingSample(self, sample(range(0,len(self)),n))

    def cosine_ranking(self, term):
        #import numpy as np
        vect = self[term] # We dont care about this norm, its constant across entire array
        dotted = mx.nd.dot(self.glove.idx_to_vec, vect)
        normed = np.linalg.norm(self.glove.idx_to_vec, axis=1)
        return np.matmul(dotted, normed)

import functools
@functools.lru_cache(maxsize=None)
def norm_of(vect):
    return vect.norm()

class CosSimilarityTo:

    def __init__(self, src_vec):
        self.src_vec = src_vec
        self.src_vec_norm = norm_of(src_vec)

    def __call__(self, dest_vec):
        sim = mx.nd.dot(self.src_vec, dest_vec) / (self.src_vec_norm * norm_of(dest_vec))
        return sim


def top_terms(embed, term, n=100, cutoff_sim_at=0.5):
    import heapq
    top_terms = []
    search_vect = embed[term]
    print("Linear scan of %s vectors" % len(embed))
    min_sim=1.01
    cutoffs=0; not_cutoffs=0; kepts=0
    sim_to=CosSimilarityTo(search_vect)
    cos_mat = embed.cosine_ranking(term)
    for cand_token, cand_vect in embed:
        sim=sim_to(cand_vect)

        if sim > cutoff_sim_at:
            not_cutoffs += 1
            if len(top_terms) <= n or sim > min_sim:
                top_terms.append( (sim, (cand_token, cand_vect)) )
                kepts += 1
                if sim < min_sim:
                    min_sim=sim
        else:
            cutoffs += 1

        if (cutoffs+not_cutoffs+kepts) % 100000 == 0:
            print("Scanned another 100k. Cutoff {} Not Cutoff {} Kept {}".format(cutoffs, not_cutoffs, kepts))

    heapq.heapify(top_terms)
    return heapq.nlargest(n, top_terms)


def random_terms(embed, num_terms=1000):
    # Get 1000 random terms
    from time import perf_counter
    sample = embed.sample(n=num_terms)
    exact_matches={}
    for term, _ in sample:
        start=perf_counter()
        print("Processing [{}]".format(term))
        top_100=top_terms(embed, term)
        exact_matches[term]=[{'token': token[1][0], 'cos': token[0]} for token in top_100]
        stop=perf_counter()
        print("Done, took %s" % (stop-start))

if __name__ == "__main__":
    embed = GloveEmbeddings()
    random_terms(embed)
