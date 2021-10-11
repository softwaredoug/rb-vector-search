"""
Glove model, based on gluon files / URL, loaded into one big numpy array
and two dicts for term / idx mapping
"""
try:
    from .download import download
except ImportError:
    from download import download

glove_model_url='https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/embeddings/glove/glove.6B.300d.npz'
glove_model_path='data/glove.6B.300d.npz'
download([glove_model_url], dest='data/')

def token_to_idx(idx_to_token):
    lookup={}
    for idx, token in enumerate(idx_to_token):
        lookup[token]=idx
    return lookup

from functools import lru_cache
@lru_cache(maxsize=1)
def glove():
    """ Return 3-tuple of
        - row x 300 matrix,
        - (row) idx->token mapping
        - token -> (row) idx mapping

        These are memoized, so call as much as you want you greedy bastard.
        Naturally all should be treated as read-only
    """
    import numpy as np
    """ Read from Gluons embedding pickle files"""
    with np.load(glove_model_path) as f:
        matrix = f['idx_to_vec']
        matrix.setflags(write=0)
        return matrix, f['idx_to_token'], token_to_idx(f['idx_to_token'])

