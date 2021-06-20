import numpy as np
import random
import json

from simulation import simulation

def main(odir):
    random.seed(19260)
    np.random.seed(817)

    params = json.load(open('2-dim-params.json', "rt"))

    U = params['U']
    A = params['A']
    W = params['W']
    T = params['T']

    seqs_list = simulation(U, A, W, T)

    print(seqs_list)

    u = []

    for i in range(len(seqs_list[0])):
        u.append((0,seqs_list[0][i]))
    for i in range(len(seqs_list[1])):
        u.append((1,seqs_list[1][i]))

    u.sort(key = lambda x: x[1])

    u = np.array(u)




    lp = 0
    sampleLen = 100

    t_seq = []
    e_seq = []

    n_train_data = 180
    n_valid_data = 12



    for i in range(256):
        t_seq.append(u[lp:lp+sampleLen,1].tolist())
        e_seq.append(u[lp:lp+sampleLen,0].tolist())
        lp = lp + 100



    with open(f'{odir}/train_time.txt', 'w') as train_time_fout:
        train_time_fout.writelines([','.join('%s'%d for d in time_sequence) + '\n' for time_sequence in t_seq[:n_train_data]])
    with open(f'{odir}/valid_time.txt', 'w') as valid_time_fout:
        valid_time_fout.writelines([','.join('%s'%d for d in time_sequence) + '\n' for time_sequence in t_seq[n_train_data:n_train_data + n_valid_data]])
    with open(f'{odir}/test_time.txt', 'w') as test_time_fout:
        test_time_fout.writelines([','.join('%s'%d for d in time_sequence) + '\n' for time_sequence in t_seq[n_train_data + n_valid_data:]])

    with open(f'{odir}/train_event.txt', 'w') as train_event_fout:
        train_event_fout.writelines([','.join('%s'%d for d in event_sequence) + '\n' for event_sequence in e_seq[:n_train_data]])
    with open(f'{odir}/valid_event.txt', 'w') as valid_event_fout:
        valid_event_fout.writelines([','.join('%s'%d for d in event_sequence) + '\n' for event_sequence in e_seq[n_train_data:n_train_data + n_valid_data]])
    with open(f'{odir}/test_event.txt', 'w') as test_event_fout:
        test_event_fout.writelines([','.join('%s'%d for d in event_sequence) + '\n' for event_sequence in e_seq[n_train_data + n_valid_data:]])

if __name__ == "__main__":
    main("data")