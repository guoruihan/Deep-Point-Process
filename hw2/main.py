import numpy as np
import random
import json
import fit
import os

from simulation import simulation

def main():
    random.seed(0)
    np.random.seed(0)

    params = json.load(open('parameters.json', "rt"))

    U = params['U']
    A = params['A']
    W = params['W']
    T = params['T']

    seqs_list = [simulation(U, A, W, T) for _ in range(10)]
    seqs = seqs_list[0]

    dirname = "result"

    fit_func = fit.fit_iterate()  # or fit_iterate
    fitted_parameters = fit_func(seqs_list, T, max_step=20, W=None, eps=1e-5, realParams=params)
    mre = fit.evaluation(params, fitted_parameters)
    fitted_parameters['mean_relative_error'] = mre
    json.dump(fitted_parameters, open(os.path.join(dirname, "fitted_parameters.json"), "wt"), indent=2)
    print("Mean relative error: {:.5f}".format(mre))

if __name__ == "__main__":
    main()