import time


def p_train(index, combiner, decision_tensor, true_assignment, queue):
    t_begin = time.perf_counter()
    combiner.train(decision_tensor, true_assignment)
    t_elapsed = time.perf_counter() - t_begin
    queue.put((index, combiner, t_elapsed))


def p_combine(index, combiner, decision_tensor, queue):
    t_begin = time.perf_counter()
    decision_matrix = combiner.combine(decision_tensor)
    t_elapsed = time.perf_counter() - t_begin
    queue.put((index, decision_matrix, t_elapsed))
