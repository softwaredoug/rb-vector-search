def recall(ground_truth, algo_result):
    tot=len(ground_truth)
    found=0
    for item in ground_truth:
        if item in algo_result[:len(ground_truth)]:
            found+=1
    return found / tot

def run_algo(token, algo):
    from glove import glove
    from exact_nn import exact_nearest_neighbors

    glove_matrix, idx_to_token, token_to_idx = glove()
    token_idx = token_to_idx[token]
    token_proj_nn, scores = algo(token_idx, glove_matrix, n=30)

    token_exact_nn, scores = exact_nearest_neighbors(token_idx, glove_matrix, n=30)
    rec = recall(ground_truth=token_exact_nn, algo_result=token_proj_nn)

    _, idx_to_token, _ = glove()
    for idx, score in zip(token_exact_nn, scores):
        if idx in token_proj_nn:
            print("HIT!-", idx_to_token[idx])
        else:
            print("MISS-", idx_to_token[idx])
    print("Recall: %s" % rec)
    return rec

def eval_algo(algo, m=100):
    """ Sample m tokens, run nearest neighbors, compute avg recall
    """
    import random
    from glove import glove
    from perf import perf_timed
    glove_matrix, idx_to_token, token_to_idx = glove()
    samples = random.sample(range(0, glove_matrix.shape[0]), m)
    rec = 0.0
    with perf_timed():
        for sample in samples:
            rec += run_algo(idx_to_token[sample], algo)
        print("Total Recall! %s" % (rec / m))
    return rec / m
