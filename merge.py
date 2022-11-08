import sys
import numpy as np


dataset = sys.argv[1]
shuffle = sys.argv[2]
FOLDS = int(sys.argv[3])



out = [[float(a.rstrip()) for a in open(f'OUTPUTS/{shuffle}_fold{i}_{dataset}.txt').read().rstrip().split('\n')] for i in range(FOLDS)]

with open(f'{shuffle}_{dataset}.txt', 'w') as f:
    for i in range(len(out[0])):
        x = np.mean([out[j][i] for j in range(FOLDS)])
        x = round(x)
        f.write(str(x) + '\n')

