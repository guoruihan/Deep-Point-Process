
from random import random
from math import pow, e, log

__ALL__ = ['simulation', 'draw_line', 'draw_qq_plot']


def simulation(U, A, W, T):
    """
    simulate a M-dimensional hawkes process
    :param U: M dimensional vector, background intensity vector
    :param A: MxM dimensional matrix, the infectivity matrix
    :param W: scalar, decay ratio
    :param T: simulate in time range [0, T]
    :return: a list of time sequence, the length of list if M
    """
    M = len(U)
    sequences = [[] for _ in range(M)]

    def calc_lambda(s, m):
        """
        calculate the m-dimension intensity on time s
        :param s: time
        :param m: dimension
        :return: the intensity
        """
        return U[m] + sum([A[m][n] * pow(e, -W * (s - hs)) for n in range(M) for hs in sequences[n]])

    s = 0.0
    while sum([len(seq) for seq in sequences]) < T:
        lambda_max = sum([calc_lambda(s, m) for m in range(M)])
        u = 1.0 - random()  # (0, 1]
        s += -log(u) / lambda_max
#        if s > T: break
        print("{:.5f} {}".format(s, " ".join(["{:3d}".format(len(seq)) for seq in sequences])))
        u = 1.0 - random()
        lambda_list = [calc_lambda(s, m) for m in range(M)]
        if u * lambda_max <= sum(lambda_list):
            k = 0
            while k < M:
                if u * lambda_max <= sum(lambda_list[:k + 1]):
                    break
                k += 1
            sequences[k].append(s)
    return sequences
