import numpy as np
import time as T
import json
import seaborn as sns

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import cartesian

import matplotlib.pyplot as plt

import random

class MHP:
    def __init__(self, alpha=None, mu=None, omega=1):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''

        if mu is None:
            mu = [0.2, 0.3 ]
        if alpha is None:
            alpha = [[0.3, 0.7], [0.1, 0.6]]
        self.data = []
        self.alpha, self.mu, self.omega = np.array(alpha), np.array(mu), omega
        self.dim = self.mu.shape[0]
        self.maxtime = 0
        self.check_stability()

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w, v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        print('Max eigenvalue: %1.5f' % me)
        if me >= 1.:
            print('(WARNING) Unstable.')

    def generate_seq(self, totnum):
        mu = self.mu
        alpha = self.alpha
        omega = self.omega
        T = totnum

        print(mu,alpha,omega,T)

        M = len(mu)

        def get_intensities():
            exps = np.array(
                [
                    np.sum(np.exp(-omega * (s - np.array(x)[np.array(x) <= s])))
                    for x in events
                ]
            )
            res = 1.0 * np.array(mu) + np.sum(np.array(alpha) * exps, axis=1)
            return res

        # initialize parameters
        self.time_seq = []
        self.event_seq = []
        events = [[] for _ in range(M)]  # track down each events
        intensities = [[] for _ in range(M)]  # track down intensities
        Ns = [0] * M
        s = 0
        cnt = 0

        # simulate M-variate Hawkes Process with Exponential Kernels
        while s < 1000 and cnt < 1200:
            if cnt == 0:
                now_intensities = 1.0 * np.array(mu)  # initial state
            else:
                now_intensities = get_intensities()

            lambda_mu = np.sum(now_intensities)
            u = random.uniform(0, 1)
            w = -np.log(u) / lambda_mu
            s = s + w
            if s > 1000:
                break

            D = random.uniform(0, 1)

            cand_intensities = (
                get_intensities()
            )  # note that we have updated s, thus intensities are not the same

            if D * lambda_mu <= np.sum(cand_intensities):
                k = 0
                while D * lambda_mu > np.sum(cand_intensities[: (k + 1)]):
                    k += 1
                Ns[k] += 1
                events[k].append(s)  # update event
                intensities[k].append(cand_intensities[k])  # update intensities
                self.time_seq.append(s)
                self.event_seq.append(k)
                cnt += 1

        self.maxtime = s

        if s > T:
            events[k].pop()
            intensities[k].pop()
            self.time_seq.pop()
            self.event_seq.pop()


        self.data = []

        # print(self.time_seq)

        for ntime, ncol in zip(self.time_seq, self.event_seq):
            self.data.append([ntime, ncol])

        #
        # return time_seq, event_seq, events, intensities

    """
    accelerated version
    some problem with the code
    temporarily put here
    """

    def generate_seq_acc(self, totnum):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below'''

        self.data = []  # clear history

        Istar = np.sum(self.mu)
        s = np.random.exponential(scale=1. / Istar)

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim),
                              1,
                              p=(self.mu / Istar))
        self.data.append([s, n0])

        # value of \lambda(t_k) where k is most recent event
        # starts with just the base rate
        lastrates = self.mu.copy()

        decIstar = False
        for i in range(totnum):
            tj, uj = self.data[-1][0], int(self.data[-1][1])

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
            else:
                # otherwise, we just had an event, so recalc Istar (inclusive of last event)
                Istar = np.sum(lastrates) + \
                        self.omega * np.sum(self.alpha[:, uj])

            # generate new event
            s += np.random.exponential(scale=1. / Istar)

            # calc rates at time s (use trick to take advantage of rates at last event)
            rates = self.mu + np.exp(-self.omega * (s - tj)) * \
                    (self.alpha[:, uj].flatten() * self.omega + lastrates - self.mu)

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            try:
                n0 = np.random.choice(np.arange(self.dim + 1), 1,
                                      p=(np.append(rates, diff) / Istar))
            except ValueError:
                # by construction this should not happen
                print('Probabilities do not sum to one.')
                self.data = np.array(self.data)
                return self.data

            if n0 < self.dim:
                self.data.append([s, n0])
                # update lastrates
                lastrates = rates.copy()
            else:
                decIstar = True

            # if past horizon, done
        self.data = np.array(self.data)
        return self.data

    # -----------
    # VISUALIZATION METHODS
    # -----------

    def get_rate(self, ct, d):
        # return rate at time ct in dimension d
        seq = np.array(self.data)
        if not np.all(ct > seq[:, 0]): seq = seq[seq[:, 0] < ct]
        return self.mu[d] + \
               np.sum([self.alpha[d, int(j)] * self.omega * np.exp(-self.omega * (ct - t)) for t, j in seq])

    def plot_rates(self, horizon=-1):

        if horizon < 0:
            horizon = np.amax(self.data[:, 0])

        # f, axarr = plt.subplots(self.dim * 2, 1, sharex='col',
        #                         gridspec_kw={'height_ratios': sum([[3, 1] for i in range(self.dim)], [])},
        #                         figsize=(8, 8))
        xs = np.linspace(0, horizon, int((horizon / 100.) * 1000))
        lst = [0 for ct in xs]
        for i in range(self.dim):
            row = i * 2

            # plot rate
            r = [self.get_rate(ct, i) for ct in xs]

            if i == 0:
                col = 'r'
            else:
                col = 'm'

            plt.plot(xs, r, col + '-')

            for j in range(len(r)):
                r[j] += lst[j]
            lst = r
            upper_r = []
            for ct, pos in zip(xs,range(len(xs))):
                upper_r.append(r[pos])
                if pos != 0:
                    if r[pos] < r[pos-1]:
                        upper_r[pos] = upper_r[pos-1]
            if i == self.dim - 1:
                plt.plot(xs, upper_r, 'g--')
            subseq = self.data[self.data[:, 1] == i][:, 0]
            if i == 0:
                col = 'bo'
            else:
                col = 'yo'
            plt.plot(subseq, np.zeros(len(subseq)) - 0.2 * i, col, alpha=0.2)

            plt.ylim([-1, np.amax(r) + (np.amax(r) / 2.)])
            # plt.set_ylabel('$\lambda(t)_{%d}$' % i, fontsize=14)
            r = []
            #
            # # plot events
            # subseq = self.data[self.data[:, 1] == i][:, 0]
            # axarr[row + 1].plot(subseq, np.zeros(len(subseq)) - 0.5, 'bo', alpha=0.2)
            # axarr[row + 1].yaxis.set_visible(False)
            #
            # axarr[row + 1].set_xlim([0, horizon])

        plt.tight_layout()
        plt.show()

    def plot_events(self, horizon=-1, showDays=False, labeled=True):
        if horizon < 0:
            horizon = np.amax(self.data[:, 0])

        fig = plt.figure(figsize=(10, 2))
        ax = plt.gca()


        for i in range(self.dim):
            subseq = self.data[self.data[:, 1] == i][:, 0]
            plt.plot(subseq, np.zeros(len(subseq)) - i, 'bo', alpha=0.2)
        # T.sleep(100)

        if showDays:
            for j in range(1, int(horizon)):
                plt.plot([j, j], [-self.dim, 1], 'k:', alpha=0.15)

        plt.show()

        if labeled:
            ax.set_yticklabels('')
            ax.set_yticks(-np.arange(0, self.dim), minor=True)
            ax.set_yticklabels([r'$e_{%d}$' % i for i in range(self.dim)], minor=True)
        else:
            ax.yaxis.set_visible(False)

        ax.set_xlim([0, horizon])
        ax.set_ylim([-self.dim, 1])
        ax.set_xlabel('Days')
        plt.tight_layout()

    def qqplot(self, pvector = None, filename=None, size=1, pointcolor='black', title=None, linecolor='red'):

        delta = []
        for i in range(len(self.data)-1):
            delta.append(self.data[i+1][0] - self.data[i][0])
        pvector = delta
        # pvector = pvector[~np.isnan(pvector)]
        # pvector = pvector[~((pvector >= 1) | (pvector <= 0))]
        pvector.sort()
        o = pvector
        e = -(np.log(1 - (np.array(range(1, (len(o) + 1))) - .5) / len(o)))
        fig, ax = plt.subplots(1, 1, figsize=(8 * int(size), 8 * int(size)), dpi=300 * (int(size) ** 2), facecolor='w')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        maxi = max(np.amax(e), np.amax(o))
        ax.set_xlim([-2, maxi + 0.1])
        ax.set_ylim([-2, maxi + 0.1])
        ax.set_ylabel('Observed -log10(' + r'$p$' + ')')
        ax.set_xlabel('Expected -log10(' + r'$p$' + ')')
        ax.scatter(e, o, c=pointcolor)

        # print(o)
        # assert(0)
        ax.plot((0, maxi), (0, maxi), linecolor)
        plt.show()

    def draw_QQ1Dim(self, Y, title="QQplot", type="expotional", num_points=200):

        Y.sort()
        # draw real data points
        N = len(Y)
        if type == "expotional":
            X = -np.log(1 - (np.linspace(1, N, N) - 0.5) / N)
        elif type == "gamma":
            X = []
        plt.scatter(X, Y, color="blue", s=8, linewidth=0.1, label="simulation data")

        # draw a line
        x = np.linspace(0, X[-1], num_points)
        y = 1.0 * x
        plt.plot(x, y, color="red", linewidth=1.0)
        plt.legend(loc="best")
        plt.title(title, fontsize="x-large")
        plt.xlabel("quantiles of expected distribution")
        plt.ylabel("quantiles of real distribution")
        plt.show()

    def draw_QQMultiDim(self,
            title,
            all_to_one=False,
            toshowDim: list = None,
    ):

        alpha = self.alpha
        mu = self.mu
        omega = self.omega
        M = len(mu)

        # calculate integral
        if all_to_one:
            intensities = []
        else:
            intensities = [[] for _ in range(M)]

        for i, t1 in enumerate(self.time_seq):
            if not all_to_one:
                dims = [self.event_seq[i]]
            else:
                dims = [i for i in range(M)]

            tmp_intensity = 0.0
            for d in dims:
                if i == 0:
                    # initial state, no other previous
                    tmp_intensity += mu[d] * t1
                    break
                elif not d in self.event_seq[:i]:
                    # never happend
                    tmp_intensity += mu[d] * t1 + np.sum(
                        [
                            1.0
                            / omega
                            * alpha[d][self.event_seq[j]]
                            * (1 - np.exp(-omega * (t1 - self.time_seq[j])))
                            for j in range(i)
                        ]
                    )
                else:
                    # happened before
                    d0 = np.where(np.array(self.event_seq[:i]) == d)[0][-1]
                    t0 = self.time_seq[d0]
                    tmp_intensity += (
                            mu[d] * (t1 - t0)
                            + np.sum([1.0 / omega * alpha[d][self.event_seq[j]]
                            * ( np.exp(-omega * (t0 - self.time_seq[j])) - np.exp(-omega * (t1 - self.time_seq[j]))
                            ) for j in range(d0) ] )

                            +

                            np.sum([1.0 / omega * alpha[d][self.event_seq[j]]
                            * (1 - np.exp(-omega * (t1 - self.time_seq[j])))
                            for j in range(d0, i) ] ) )

            if all_to_one:
                assert len(dims) == M
                intensities.append(tmp_intensity)
                with open("tmp.csv", "w") as f:
                    json.dump(intensities, f)
            else:
                assert len(dims) == 1
                intensities[dims[0]].append(tmp_intensity)

        if all_to_one:
            self.draw_QQ1Dim(intensities, "{}-alldim QQ plot".format(title))

        else:
            if toshowDim is None:
                toshowDim = range(min(len(mu), 2))

            for d in toshowDim:
                self.draw_QQ1Dim(intensities[d], "{}-dim-{} QQ plot".format(title, d))

        # if title is not None:
        #     ax.set_title(title)
        # if filename is not None:
        #     plt.savefig(filename, dpi=300, bbox_inches='tight')
        #     plt.close()
        # else:
        #     return fig, ax
# def spectral_radius(M):
#     a,b=np.linalg.eig(M) #a为特征值集合，b为特征值向量
#     return np.max(np.abs(a))
# 
# spec = []
# 
# def gen_matrix(dim):
#     matrix = [[random.random() for i in range(dim)] for j in range(dim)]
#     while(spectral_radius(matrix) > 0.9):
#         matrix = [[random.random()/10 for i in range(dim)] for j in range(dim)]
#         spec.append(spectral_radius(matrix))
#         if len(spec) > 1000:
#             break
#     return matrix
# 
# m = np.array([0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, ])
# a = gen_matrix(10)
# 
# print(spectral_radius(a))
# 
# 
# 
# assert(0)
#
# w = 1



# P = MHP(mu=m, alpha=a, omega=w)
P = MHP()
P.mu =[random.random() for i in range(10)]
P.alpha = []
for i in range(10):
    nl = []
    for j in range(10):
        if i == j:
            nl.append(0.8 + 0.2 * random.random())
        else:
            nl.append(0.1 * random.random())
    P.alpha.append(nl)
P.generate_seq(1000)
# P.plot_events()
# plt.show()
# P.plot_rates()
P.draw_QQMultiDim("10dim")