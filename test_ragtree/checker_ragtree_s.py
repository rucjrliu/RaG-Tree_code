import csv
import numpy as np
import sys

if __name__ == '__main__':
    # print(sys.argv[1:])
    # exit()
    usage = 'Usage: python checker_selrandom_rtreehnsw_s.py <dataset> <ef> [a|d] [o|d]'
    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    dataset_name = sys.argv[1]
    ef = sys.argv[2]
    if dataset_name in ('a', 'd', 'o'):
        print('Error: dataset name is required; old format <ef> [a|d] [o|d] is not supported.')
        print(usage)
        sys.exit(1)
    if not ef.isdigit():
        print('Error: ef must be an integer; expected argument order is <dataset> <ef> [a|d] [o|d].')
        print(usage)
        sys.exit(1)

    # A = []
    A_name = '../data/{}/{}_selrandom_gt.npz'.format(dataset_name, dataset_name)
    A = np.load(A_name)['gt_indices']
    k = A.shape[1]
    # fileinA = open(A_name, 'r', encoding='utf-8')
    # s = fileinA.readlines()
    # fileinA.close()
    # for i in range(1, len(s)):
    #     s_i = s[i].strip('\n').strip(' ')
    #     if s_i == '':
    #         A.append([])
    #     else:
    #         A.append(list(map(int, s_i.split(' '))))
    # for i in A:
    #     i.sort()

    B = []
    flag_ada = 'a' if len(sys.argv) >= 4 and sys.argv[3][0] == 'a' else 'd'
    flag_opt = 'o' if len(sys.argv) >= 5 and sys.argv[4][0] == 'o' else 'd'
    B_name = '../output/result_ragtree_s_{}_ef{}_{}{}_selrandom.log'.format(dataset_name, ef, flag_ada, flag_opt)
    print('checker %s:' %(B_name))
    fileinB = open(B_name, 'r', encoding='utf-8')
    s = fileinB.readlines()
    fileinB.close()
    for i in range(1, len(s)):
        s_i = s[i].strip('\n').strip(' ')
        if s_i == '':
            B.append([])
        else:
            B.append(list(map(int, s_i.split(' '))))

    recalls = []
    assert len(A) == len(B)
    for i in range(len(A)):
        recall_num = 0
        Aians = set(A[i])
        Bires = set(B[i][0 : min(len(B[i]), k)])
        # print(Aians, Bires)
        for j in Bires:
            if j in Aians:
                recall_num += 1
        # print('Query{} recall={}'.format(i, recall_num/k))
        recalls.append(recall_num / k)
    # print(recalls)
    print('Avg Recall =', np.mean(recalls))
