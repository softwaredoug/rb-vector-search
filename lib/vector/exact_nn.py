import numpy as np

if __name__ != "__main__":
    from .perf import perf_timed
    from .glove import glove
    from .download import download
else:
    from perf import perf_timed
    from glove import glove
    from download import download


def exact_nearest_neighbors(row, matrix, n=100):
    """ nth nearest neighbors as array
        with indices of nearest neighbors"""
    token_vect = matrix[row]
    if exact_nearest_neighbors.normed is None:
        exact_nearest_neighbors.normed = np.linalg.norm(matrix, axis=1)

    dotted = np.dot(matrix, token_vect)
    nn = np.divide(dotted, exact_nearest_neighbors.normed)
    top_n = np.argpartition(-nn, n)[:n]
    return top_n, nn[top_n]

exact_nearest_neighbors.normed=None

def print_glove_nearest_neighbors(token):
    glove_matrix, idx_to_token, token_to_idx = glove()
    token_idx = token_to_idx[token]
    nn, scores = exact_nearest_neighbors(token_idx, glove_matrix, n=30)
    _, idx_to_token, _ = glove()
    for idx, score in zip(nn, scores):
        print(score, idx_to_token[idx])


def main(tokens):
    download(['https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/embeddings/glove/glove.6B.300d.npz'])
    for token in tokens:
        print("==========================")
        print("%s nn:" % token)
        with perf_timed():
            print_glove_nearest_neighbors(token)

if __name__ == "__main__":
    from sys import argv
    main(argv[1:])
