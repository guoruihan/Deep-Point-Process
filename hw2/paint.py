import bisect
import copy
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import pow, e, log

def draw_line(seqs, file):
    """
    draw the events in a time line
    :param seqs: list of sequence
    :param file: None or string, if file is None, draw and show on screen, otherwise save the image using file as filename
    """
    fig, ax = plt.subplots()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(seqs))))
    for index, seq in enumerate(seqs):
        color = next(colors)
        label = str(index)
        seq_filtered = list(filter(lambda x: x < 200.0, seq))
        ax.scatter(x=seq_filtered, y=np.zeros_like(seq_filtered), s=10, c=[color], label=label, alpha=1.0)
    ax.legend(loc='upper center', mode='expand', ncol=len(seqs))
    ax.grid(True)
    ax.get_yaxis().set_visible(False)
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()
    plt.close()


def draw_qq_plot(parameters, seqs, file):
    """
    draw the quantile-quantile plot
    :param parameters: parameters of the hawkes process
    :param seqs: list of sequence
    :param file: None or string, if file is None, draw and show on screen, otherwise save the image using file as filename
    """
    W = parameters['W']
    U = parameters['U']
    A = parameters['A']
    samples = []
    seqs = copy.deepcopy(seqs)
    for seq in seqs:
        seq.insert(0, 0.0)
    M = len(seqs)
    R = [[None for _ in range(M)] for __ in range(M)]
    for m in range(M):
        for n in range(M):
            R[m][n] = [0.0 for _ in range(len(seqs[m]))]
            R[m][n][0] = 0.0
            for k in range(1, len(seqs[m])):
                R[m][n][k] = pow(e, -W * (seqs[m][k] - seqs[m][k-1])) * R[m][n][k-1]
                begin = bisect.bisect_left(seqs[n], seqs[m][k-1])
                end = bisect.bisect_right(seqs[n], seqs[m][k])
                for i in range(begin, end):
                    R[m][n][k] += pow(e, -W * (seqs[m][k] - seqs[n][i]))
    for m in range(M):
        for k in range(1, len(seqs[m])):
            s = U[m] * (seqs[m][k] - seqs[m][k-1])
            for n in range(M):
                s += (A[m][n] / W) * (1.0 - pow(e, -W * (seqs[m][k] - seqs[m][k-1]))) * R[m][n][k-1]
                begin = bisect.bisect_left(seqs[n], seqs[m][k-1])
                end = bisect.bisect_right(seqs[n], seqs[m][k])
                for i in range(begin, end):
                    s += (A[m][n] / W) * (1.0 - pow(e, -W * (seqs[m][k] - seqs[n][i])))
            samples.append(s)
    scipy.stats.probplot(x=samples, dist=scipy.stats.expon(), fit=False, plot=plt)
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()
    plt.close()

    plt.hist(samples, bins=30)
    plt.savefig("result/hist.svg")