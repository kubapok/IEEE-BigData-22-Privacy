import pandas as pd
from pathlib import Path
import pickle



def run(DATASET):
    r = pd.read_csv('../../BigDataCup2022/submission_template.csv')

    p2_enc = sorted(Path(f'../../BigDataCup2022/{DATASET}/test/enc/').glob('*'))

    p2_input = sorted(Path(f'../../BigDataCup2022/{DATASET}/test/input/').glob('*'))

    r_d = {'enc_path': p2_enc, 'input_path':p2_input}

    r = pd.DataFrame(data = r_d)

    r['enc_id'] = r['enc_path'].apply(lambda x: x.stem)

    r['input_id'] = r['input_path'].apply(lambda x: x.stem)

    with open(f'CACHED/{DATASET}_X_test.pickle', 'wb') as f:
        pickle.dump(r,f)

run('S1')
run('S2')
run('S3')
