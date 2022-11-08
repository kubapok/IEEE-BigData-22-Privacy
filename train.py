#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
from itertools import product
import cv2
from xgboost import XGBClassifier
import numpy as np
import copy
from tqdm import tqdm
import sys
from catboost import Pool, CatBoostClassifier
import random
import pandas as pd

# In[2]:


DATASET = sys.argv[1]
SHUFFLES = sys.argv[2]
fold = sys.argv[3]
NOTEBOOK_NUMBER = 'NOTEBOOK'
NOTEBOOK_NAME =  SHUFFLES
TEST = 0
task_type='CPU'
NJOBS=15

SHUFFLES = int(SHUFFLES)
# 0- only positives
# 1 - 1 part negatives; 1 part positives
# 2 - 2 part negatives; 2 part positives
# etc.

# In[3]:


#DATASET = 'S1'

# ## dataset loading


with open(f'CACHED/{DATASET}_fold{fold}_X_train.pickle', 'rb') as f:
    X_train = pickle.load(f)
with open(f'CACHED/{DATASET}_fold{fold}_y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)
with open(f'CACHED/{DATASET}_fold{fold}_X_dev.pickle','rb') as f:
    X_dev = pickle.load(f)
with open(f'CACHED/{DATASET}_fold{fold}_y_dev.pickle','rb') as f:
    y_dev = pickle.load(f)
with open(f'CACHED/{DATASET}_X_test.pickle','rb') as f:
    X_test = pickle.load(f)

# ## smaller train dataset

TEST_LEN = 10
if TEST:
    X_train = X_train.head(TEST_LEN)
    X_dev = X_dev.head(TEST_LEN)
    X_test = X_test.head(TEST_LEN)
    
    
    y_train = y_train.head(TEST_LEN)
    y_dev = y_dev.head(TEST_LEN)


# ## feature extraction



# keep only positives
X_train = (X_train[X_train['input_id'] == X_train['enc_id']])
X_train = X_train.drop_duplicates('input_id')

enc_path = list(X_train['enc_path'])
input_path = list(X_train['input_path'])
random.seed(124)

encoded_list = []
input_list = []
for i in range(SHUFFLES):
    encoded_list += copy.deepcopy(enc_path)
    input_list += copy.deepcopy(input_path)

for i in range(SHUFFLES):
    enc_shuffle = copy.deepcopy(enc_path)
    random.shuffle(enc_shuffle)
    encoded_list += enc_shuffle
    input_list += copy.deepcopy(input_path)

r_d = {'enc_path': encoded_list, 'input_path': input_list}
r = pd.DataFrame(data=r_d)
r['enc_id'] = r['enc_path'].apply(lambda x: x.stem)
r['input_id'] = r['input_path'].apply(lambda x: x.stem)
r['is_pair'] = (r['enc_id'] == r['input_id']).astype(int)

y_train = r['is_pair']
del r['is_pair']
X_train = r



def process_file(path):
    path = str(path)
    features = []
    x_color = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    x_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


    features += [x_color.mean(), x_color.min(), x_color.max(), x_color.var(), x_color.sum()]
    features += [np.percentile(x_color, p) for p in np.arange(0.05, 1+0.05, 0.05)]
     
    for color in range(3): # rgb
        
        x2 = x_color[:, :, color].reshape(-1,)
        
        color_hist = [np.count_nonzero(x2 == i) for i in range(256)]
        features += copy.deepcopy(color_hist)
        features += [x2.mean(), x2.min(), x2.max(), x2.var(), x2.sum()]
        features += [np.percentile(x2, p) for p in np.arange(0.05, 1+0.05, 0.05)]

        features += [np.mean(color_hist), np.min(color_hist), np.max(color_hist), np.var(color_hist), np.sum(color_hist)]
        features += [np.percentile(color_hist, p) for p in np.arange(0.05, 1+0.05, 0.05)]



    gray_hist = np.histogram(x_gray, 255)[0].tolist()
    features += copy.deepcopy(gray_hist)
    features += [np.mean(gray_hist), np.min(gray_hist), np.max(gray_hist), np.var(gray_hist), np.sum(gray_hist)]
    features += [np.percentile(gray_hist, p) for p in np.arange(0.05, 1+0.05, 0.05)]

    features += [x_gray.mean(), x_gray.min(), x_gray.max(), x_gray.var(), x_gray.sum()]
    features += [np.percentile(x_gray, p) for p in np.arange(0.05, 1+0.05, 0.05)]





    for INTERVAL_LEN in (32, 64):

        intervals = range(0,255,INTERVAL_LEN)
        all_comb =  product(intervals, intervals, intervals)

        for a,b,c in all_comb:
            features += [np.sum(    (a < x_color[:, :, 0]) & (x_color[:, :, 0] < a+INTERVAL_LEN)
                              & (b < x_color[:, :, 1]) & (x_color[:, :, 1] < b+INTERVAL_LEN)
                              & (c < x_color[:, :, 2]) & (x_color[:, :, 2] < c+INTERVAL_LEN))]

        for a in range(0,255, INTERVAL_LEN):
            features += [np.sum(    (a < x_gray) & (x_gray < a+INTERVAL_LEN) ) ]



    return features

FEATURE_NUMBER = 1836



def process_dataset(dataset):
    input_features = np.zeros((len(dataset), FEATURE_NUMBER)) - 1
    enc_features = np.zeros((len(dataset), FEATURE_NUMBER)) - 1

    idx = 0
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        if row['input_path'] not in processed_cache:
            processed_cache[row['input_path']] = process_file(str(row['input_path']))
        input_features[idx] = processed_cache[row['input_path']]

        if row['enc_path'] not in processed_cache:
            processed_cache[row['enc_path']] = process_file(str(row['enc_path']))
        enc_features[idx] = processed_cache[row['enc_path']]
        idx += 1
    return input_features, enc_features


CACHE_NAME = f'{DATASET}_{NOTEBOOK_NUMBER}_processed_cache.pickle'
print(f'CACHE_NAME:{CACHE_NAME}')
try:
    with open(CACHE_NAME, 'rb') as f:
        processed_cache = pickle.load(f)
except FileNotFoundError:
    processed_cache = dict()


EPS = 0.0001
a, b = process_dataset(X_train)
X_train_features = np.concatenate( (a, b, a-b, b-a, a+b, a/(b+EPS), b/(a+EPS)), axis=1)

a, b = process_dataset(X_dev)
X_dev_features = np.concatenate((a, b, a-b, b-a, a+b, a/(b+EPS), b/(a+EPS)), axis=1)

#import pdb; pdb.set_trace()
a, b = process_dataset(X_test)
X_test_features = np.concatenate((a, b, a-b, b-a, a+b, a/(b+EPS), b/(a+EPS)), axis=1)

#with open(CACHE_NAME, 'wb') as f:
#    pickle.dump(processed_cache, f)

# ### training a model

model = CatBoostClassifier(n_estimators=5000, eval_metric="Accuracy", random_seed=int(fold), task_type=task_type,thread_count=NJOBS)


# In[ ]:


model.fit(X_train_features, y_train, eval_set=(X_dev_features, y_dev), early_stopping_rounds=1000, use_best_model=True)

print('best ntree:', model.get_best_iteration())
print('best score:', model.get_best_score()) 





# ## predictions

train_predictions = model.predict_proba(X_train_features)


dev_predictionsa = model.predict(X_dev_features)

print(sum(dev_predictionsa == y_dev)/len(y_dev))


dev_predictions = model.predict_proba(X_dev_features)[:,1]

# ## test predictions 

test_predictions = model.predict_proba(X_test_features)[:,1]


with open(f'OUTPUTS/{NOTEBOOK_NAME}_fold{fold}_{DATASET}.txt', 'w') as f:
    for i in test_predictions:
        to_write = str(i) + '\n'
        f.write(to_write)

# ### dummy S3

with open(f'OUTPUTS/{NOTEBOOK_NAME}_S3.txt','w') as f:
    for i in range(10_000):
        to_write = str(0) + '\n'
        f.write(to_write)
