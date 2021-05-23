import numpy as np
import time as T
import seaborn as sns

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import cartesian

import matplotlib.pyplot as plt

import random

class MHP:
    def __init__(self, alpha=[[0.3]], mu=[0.2], omega=0.5):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''

        self.data = []
        self.alpha, self.mu, self.omega = np.array(alpha), np.array(mu), omega
        self.dim = self.mu.shape[0]
        self.check_stability()

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w, v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        print('Max eigenvalue: %1.5f' % me)
        if me >= 1.:
            print('(WARNING) Unstable.')

    def generate_seq(self, totnum):
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
    # EM LEARNING
    # -----------

    def EM(self, Ahat, mhat, omega, seq=[], smx=None, tmx=None, regularize=False,
           Tm=-1, maxiter=100, epsilon=0.01, verbose=True):
        '''implements MAP EM. Optional to regularize with `smx` and `tmx` matrix (shape=(dim,dim)).
        In general, the `tmx` matrix is a pseudocount of parent events from column j,
        and the `smx` matrix is a pseudocount of child events from column j -> i, 
        however, for more details/usage see https://stmorse.github.io/docs/orc-thesis.pdf'''

        # if no sequence passed, uses class instance data
        if len(seq) == 0:
            seq = self.data

        N = len(seq)
        dim = mhat.shape[0]
        Tm = float(seq[-1, 0]) if Tm < 0 else float(Tm)
        sequ = seq[:, 1].astype(int)

        p_ii = np.random.uniform(0.01, 0.99, size=N)
        p_ij = np.random.uniform(0.01, 0.99, size=(N, N))

        # PRECOMPUTATIONS

        # diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        diffs = pairwise_distances(np.array([seq[:, 0]]).T, metric='euclidean')
        diffs[np.triu_indices(N)] = 0

        # kern[i,j] = omega*np.exp(-omega*diffs[i,j])
        kern = omega * np.exp(-omega * diffs)

        colidx = np.tile(sequ.reshape((1, N)), (N, 1))
        rowidx = np.tile(sequ.reshape((N, 1)), (1, N))

        # approx of Gt sum in a_{uu'} denom
        seqcnts = np.array([len(np.where(sequ == i)[0]) for i in range(dim)])
        seqcnts = np.tile(seqcnts, (dim, 1))

        # returns sum of all pmat vals where u_i=a, u_j=b
        # *IF* pmat upper tri set to zero, this is 
        # \sum_{u_i=u}\sum_{u_j=u', j<i} p_{ij}
        def sum_pij(a, b):
            c = cartesian([np.where(seq[:, 1] == int(a))[0], np.where(seq[:, 1] == int(b))[0]])
            return np.sum(p_ij[c[:, 0], c[:, 1]])

        vp = np.vectorize(sum_pij)

        # \int_0^t g(t') dt' with g(t)=we^{-wt}
        # def G(t): return 1 - np.exp(-omega * t)
        #   vg = np.vectorize(G)
        # Gdenom = np.array([np.sum(vg(diffs[-1,np.where(seq[:,1]==i)])) for i in range(dim)])

        k = 0
        old_LL = -10000
        START = T.time()
        while k < maxiter:
            Auu = Ahat[rowidx, colidx]
            ag = np.multiply(Auu, kern)
            ag[np.triu_indices(N)] = 0

            # compute m_{u_i}
            mu = mhat[sequ]

            # compute total rates of u_i at time i
            rates = mu + np.sum(ag, axis=1)

            # compute matrix of p_ii and p_ij  (keep separate for later computations)
            p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1, N)))
            p_ii = np.divide(mu, rates)

            # compute mhat:  mhat_u = (\sum_{u_i=u} p_ii) / T
            mhat = np.array([np.sum(p_ii[np.where(seq[:, 1] == i)]) \
                             for i in range(dim)]) / Tm

            # ahat_{u,u'} = (\sum_{u_i=u}\sum_{u_j=u', j<i} p_ij) / \sum_{u_j=u'} G(T-t_j)
            # approximate with G(T-T_j) = 1
            if regularize:
                Ahat = np.divide(np.fromfunction(lambda i, j: vp(i, j), (dim, dim)) + (smx - 1),
                                 seqcnts + tmx)
            else:
                Ahat = np.divide(np.fromfunction(lambda i, j: vp(i, j), (dim, dim)),
                                 seqcnts)

            if k % 10 == 0:
                try:
                    term1 = np.sum(np.log(rates))
                except:
                    print('Log error!')
                term2 = Tm * np.sum(mhat)
                term3 = np.sum(np.sum(Ahat[u, int(seq[j, 1])] for j in range(N)) for u in range(dim))
                # new_LL = (1./N) * (term1 - term2 - term3)
                new_LL = (1. / N) * (term1 - term3)
                if abs(new_LL - old_LL) <= epsilon:
                    if verbose:
                        print('Reached stopping criterion. (Old: %1.3f New: %1.3f)' % (old_LL, new_LL))
                    return Ahat, mhat
                if verbose:
                    print('After ITER %d (old: %1.3f new: %1.3f)' % (k, old_LL, new_LL))
                    print(' terms %1.4f, %1.4f, %1.4f' % (term1, term2, term3))

                old_LL = new_LL

            k += 1

        if verbose:
            print('Reached max iter (%d).' % maxiter)

        self.Ahat = Ahat
        self.mhat = mhat
        return Ahat, mhat

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
        o = -(np.log10(pvector))
        e = -(np.log10((np.array(range(1, (len(o) + 1))) - .5) / len(o)))
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
P.generate_seq(1000)
# P.plot_events()
# plt.show()
# P.plot_rates()
P.qqplot()