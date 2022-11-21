import pandas as pd
from pathlib import Path
import random
import copy
import sklearn
import sklearn.model_selection
import pickle
import sys

#DATADIR = '/mnt/gpu_data1/kubapok/bigdata2022challenge-encryption/bigdatachallange/BigDataCup2022'
DATADIR=sys.argv[1]
FOLDS=int(sys.argv[2])


def run(DATASET, fold):
    r = pd.read_csv(DATADIR + '/submission_template.csv')

    p2_enc = sorted(Path(DATADIR + f'/{DATASET}/train/enc/').glob('*'))

    p2_input = sorted(Path(DATADIR + f'/{DATASET}/train/input/').glob('*'))

    random.seed(fold)

    p2_input_shuffle = copy.deepcopy(p2_input)

    random.shuffle(p2_input_shuffle)

    r_d = {'enc_path': p2_enc + p2_enc, 'input_path':p2_input + p2_input_shuffle}

    r = pd.DataFrame(data = r_d)

    r['enc_id'] = r['enc_path'].apply(lambda x: x.stem)

    r['input_id'] = r['input_path'].apply(lambda x: x.stem)

    r['is_pair'] = (r['enc_id'] == r['input_id']).astype(int)

    X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(r.loc[:, r. columns != 'is_pair'], r['is_pair'], test_size = 0.30, random_state = fold)

    with open(f'CACHED/{DATASET}_fold{fold}_X_train.pickle', 'wb') as f:
        pickle.dump(X_train,f)

    with open(f'CACHED/{DATASET}_fold{fold}_X_dev.pickle', 'wb') as f:
        pickle.dump(X_dev,f)

    with open(f'CACHED/{DATASET}_fold{fold}_y_train.pickle', 'wb') as f:
        pickle.dump(y_train,f)

    with open(f'CACHED/{DATASET}_fold{fold}_y_dev.pickle', 'wb') as f:
        pickle.dump(y_dev,f)

for i in range(FOLDS):
    run('S1', i)

for i in range(FOLDS):
    run('S2', i)
