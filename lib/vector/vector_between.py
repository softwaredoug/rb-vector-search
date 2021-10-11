import numpy as np

def vector_between(vec1, vec2):
    # Find two positive of vec1 and vec2
    idxs = np.array(range(0,vec1.shape[0]))
    both_positive = np.intersect1d(idxs[v2>0], idxs[v1>0])

    v = np.random.nand( vec1.shape )
