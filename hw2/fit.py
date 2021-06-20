import math
import itertools
import numpy as np
from numpy.linalg import norm

def relative_error(x, y):
    up = np.abs(x - y)
    down = x + y / 2.0
    down = np.where(down <= 0.0, np.full_like(down, 1.0), down)
    return np.mean(up / down)


def flatten_concat(arr_list):
    return np.concatenate([arr.flatten() for arr in arr_list])


def evaluation(real_parameters, fitted_parameters):
    """
    return the mean relative error
    """
    U1, U2 = real_parameters['U'], fitted_parameters['U']
    A1, A2 = real_parameters['A'], fitted_parameters['A']
    w1, w2 = real_parameters['W'], fitted_parameters['W']
    X1, X2 = [], []
    X1.extend(U1)
    X2.extend(U2)
    for a1, a2 in zip(A1, A2):
        X1.extend(a1)
        X2.extend(a2)
    if w1 != w2:
        X1.append(w1)
        X2.append(w2)
    X1, X2 = np.array(X1), np.array(X2)
    down = X1.copy()
    down[np.where(down == 0)] = 1.0
    return np.mean(np.abs(X1-X2) / down)


def fit_step(step, T, e, U, A, W, w_fixed, realParams=None):
    N = len(e)
    M = U.shape[0]
    p = np.zeros((N, N))
    old_U = np.copy(U)
    old_A = np.copy(A)
    old_w = np.copy(W)

    for i in range(N):
        for j in range(i):
            p[i, j] = A[e[j][1], e[i][1]] * pow(math.e, -W * (e[i][0] - e[j][0]))
        p[i, i] = U[e[i][1]]
        p[i] = p[i] / np.sum(p[i])
    for d in range(M):
        U[d] = sum([p[i, i] for i in range(N) if e[i][1] == d]) / T
    for du in range(M):
        for dv in range(M):
            up, down = 0.0, 0.0
            for i in range(N):
                if e[i][1] != dv: continue
                for j in range(i):
                    if e[j][1] != du: continue
                    up += p[i, j]
            for j in range(N):
                if e[j][1] != du: continue
                down += (1.0 - pow(math.e, -old_w * (T - e[j][0]))) / old_w
            if down == 0.0:
                A[dv, du] = 0.0
            else:
                A[dv, du] = up / down
    if not w_fixed:
        up, down = 0.0, 0.0
        for i in range(N):
            for j in range(i):
                pij = p[i, j]
                up += pij
                down += (e[i][0] - e[j][0]) * pij
        W = up / down
    else:
        W = old_w
    U_error = relative_error(old_U, U)
    A_error = relative_error(old_A, A)
    w_error = relative_error(old_w, W)
    dist = relative_error(flatten_concat([old_U, old_A, old_w]), flatten_concat([U, A, W]))
    eva = evaluation(realParams, {'U': U, 'A': A, 'W': W}) if realParams is not None else "None"
    print("\rStep  {} EVA {}  ALL {:.7f}  U {:.7f}  A {:.7f}  W {:.7f}".format(step, eva, dist, U_error, A_error, w_error), end="")
    return U, A, W


def fit_iterate(seqs_list, T, W=None, max_step=1000, eps=1e-5, realParams=None):
    e_list = []
    for seqs in seqs_list:
        e = []
        for index, seq in enumerate(seqs):
            e.extend(zip(seq, itertools.repeat(index)))
        e = sorted(e, key=lambda event: event[0])
        e_list.append(e)

    w_fixed = W is not None
    M = len(seqs_list[0])
    U = np.random.rand(M)
    A = np.random.rand(M, M)
    if not w_fixed:
        W = np.random.rand()
    for step, e in zip(range(max_step), itertools.cycle(e_list)):
        U, A, W = fit_step(step, T, e, U, A, W, w_fixed, realParams)
    return {'U': U, 'A': A, 'W': W}

