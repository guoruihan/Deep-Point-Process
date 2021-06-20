import itertools
import numpy as np
import matplotlib.pyplot as plt

__ALL__ = ['fit', 'fit_iterate']


def relative_error(x, y):
    up = np.abs(x - y)
    down = x + y / 2.0
    down = np.where(down <= 0.0, np.full_like(down, 1.0), down)
    return np.mean(up / down)


def flatten_concat(arr_list):
    return np.concatenate([arr.flatten() for arr in arr_list])


def evaluation(real_parameters, fitted_parameters):
    def compress(params):
        U = params['U']
        A = params['A']
        W = params['W']
        V = []
        V.extend(U)
        for a in A:
            V.extend(a)
        V.append(W)
        return np.array(V)

    V1 = compress(real_parameters)
    V2 = compress(fitted_parameters)

    def calcup():
        return np.abs(V1-V2)

    def calcdown():
        down = V1.copy()
        down[np.where(down == 0)] = 1.0
        return down

    up = calcup()
    down = calcdown()
    return np.mean(up / down)

def fit_single(seqs, T, W=None, max_step=30, eps=1e-5, realParams=None):
    """
    inference the multi-hawkes point process parameters
    :param seqs: the list of event sequences, M = len(seqs) is the dimension of the process
    :param T: the data was sampled from time interval [0, T]
    :param W: when W is None, we inference W, otherwise we regard W is known
    :param max_step: the maximum number of steps
    :param eps: the epsilon, when the 2-norm of change is less or equal to epsilon, stop iteration
    :return: parameters, {'U': U, 'A', A, 'W': W}, where U is the background intensity
        and A is the infectivity matrix. A[n][m] is the infectivity of n to m
    """
    T = max([max(seq) for seq in seqs])
    print(T)
    M = len(seqs)
    w_known = W is not None
    U = np.random.uniform(0, 0.1, size=M)
    A = np.random.uniform(0, 0.1, size=(M, M))
    if not w_known:
        W = np.random.uniform(0, 1, size=1)

    e = []
    for index, seq in enumerate(seqs):
        e.extend(zip(seq, itertools.repeat(index)))
    e = sorted(e, key=lambda event: event[0])
    N = len(e)
    p = np.zeros((N, N))

    relErr = []

    for step in range(max_step):

        old_U = np.copy(U)
        old_A = np.copy(A)
        old_W = np.copy(W)

        # update p
        for i in range(N):
            for j in range(i):
                p[i, j] = old_A[e[i][1], e[j][1]] * np.exp(-W * (e[i][0] - e[j][0]))
            p[i, i] = old_U[e[i][1]]
            p[i] = p[i] / np.sum(p[i])

        # update U
        for d in range(M):
            U[d] = sum([p[i, i] for i in range(N) if e[i][1] == d]) / T

        # update A
        for du in range(M):
            for dv in range(M):
                up, down = 0.0, 0.0
                for i in range(N):
                    if e[i][1] != du: continue
                    for j in range(i):
                        if e[j][1] != dv: continue
                        up += p[i, j]
                for j in range(N):
                    if e[j][1] != dv: continue
                    down += (1.0 - np.exp(-old_W * (T - e[j][0]))) / old_W
                A[du, dv] = up / down

        # update W
        if not w_known:
            up, down = 0.0, 0.0
            for i in range(N):
                for j in range(i):
                    pij = p[i, j]
                    up += pij
                    down += (e[i][0] - e[j][0]) * pij
            W = up / down
        else:
            W = old_W

        eva = evaluation(realParams, {'U': U, 'A': A, 'W': W})

        relErr.append(eva)

        print("\nStep  {} EVA {}".format(step, eva), end="")
    print()

    plt.plot(range(max_step), relErr)
    plt.show()

    return {'U': U, 'A': A, 'W': W}


def fit(seqs_list, T, W=None, max_step=1000, eps=1e-5, realParams=None):
    """
    inference the multi-hawkes point process parameters
    :param seqs_list: the list of the list of event sequences, M = len(seqs_list[0]) is the dimension of the process
    :param T: the data was sampled from time interval [0, T]
    :param W: when W is None, we inference W, otherwise we regard W is known
    :param max_step: the maximum number of steps
    :param eps: the epsilon, when the 2-norm of change is less or equal to epsilon, stop iteration
    :return: parameters, {'U': U, 'A', A, 'W': W}, where U is the background intensity
        and A is the infectivity matrix. A[n][m] is the infectivity of n to m
    """
    U_list = []
    A_list = []
    w_list = []
    for index, seqs in enumerate(seqs_list):
        print("Sequence {} / {}".format(index, len(seqs_list)))
        params = fit_single(seqs, T, W, max_step, eps, realParams)
        U_list.append(params['U'])
        A_list.append(params['A'])
        w_list.append(params['W'])
    U = np.mean(U_list, axis=0)
    A = np.mean(A_list, axis=0)
    W = np.mean(w_list, axis=0)
    return {'U': U.tolist(), 'A': A.tolist(), 'W': W.tolist()}