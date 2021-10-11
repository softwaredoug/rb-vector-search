import numpy as np
from glove import glove

class AdaptRandomProjections:
    """ Adaptable random projections, similar to Annoy, where
        the projections are selected based on two vectors from the data """

    def __init__(self, matrix, num_vects=1000):
        cols=matrix.shape[1]
        self.projs=np.random.rand( cols, num_vects  )
        #self.projs -= 0.5
        self.hashed=np.matmul(matrix, self.projs)
        self.hashed[self.hashed >= 0] = 1
        self.hashed[self.hashed <  0] = -1
        self.hashed=self.hashed.astype(int)

    def __call__(self, row, matrix, n=100):
        # should just do this once

        # Dot with row to get sim score

        nn = np.dot(self.hashed, self.hashed[row])

        # Rank top N rows
        top_n = np.argpartition(-nn, n)[:n]
        return top_n, nn[top_n]


def print_glove_nearest_neighbors(token, rand_proj):
    glove_matrix, idx_to_token, token_to_idx = glove()
    token_idx = token_to_idx[token]
    nn, scores = rand_proj(token_idx, glove_matrix, n=30)
    _, idx_to_token, _ = glove()
    for idx, score in zip(nn, scores):
        print(score, idx_to_token[idx])

if __name__ == "__main__":
    from sys import argv
    from perf import perf_timed
    tokens=argv[1:]
    glove_matrix, idx_to_token, token_to_idx = glove()
    rand_proj=RandomProjections(glove_matrix)
    for token in tokens:
        print("==========================")
        print("%s nn:" % token)
        with perf_timed():
            print_glove_nearest_neighbors(token, rand_proj)


