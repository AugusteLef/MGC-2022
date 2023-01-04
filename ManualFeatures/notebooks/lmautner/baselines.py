#!/usr/bin/env python
# coding: utf-8

# ### Imports & Preparation

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import torch
import time
import os


# In[2]:


on_cluster = True


# In[3]:


import sys

sys.path.insert(1, '/cluster/home/lmautner/music-classification-DL22/' if on_cluster else '/home/lmautner/music-classification-DL22/')


# In[4]:


from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.datamodules.components.musical_features_dataset import MusicalFeaturesDataset


def read_data(data_dir: str) -> List[MusicalFeaturesDataset]:
    features = pd.read_csv(f'{data_dir}/fma_metadata/features.csv', index_col=0, header=[0, 1, 2])

    tracks = pd.read_csv(f'{data_dir}/fma_metadata/tracks.csv', index_col=0, header=[0, 1])
    tracks['track', 'genre_top'] = tracks['track', 'genre_top'].astype('category')
    tracks = tracks[[('track', 'genre_top'), ('set', 'split')]].dropna()

    features = features.loc[[idx for idx in features.index if idx in tracks.index]]
    tracks = tracks.loc[[idx for idx in tracks.index if idx in features.index]]

    features = features.sort_index()
    tracks = tracks.sort_index()

    index_train = tracks.index[(tracks['set', 'split'] == 'training') | (tracks['set', 'split'] == 'validation')]
    index_test = tracks.index[(tracks['set', 'split'] == 'test')]

    datasets = []
    for idx in [index_train, index_test]:
        tracks_idx = tracks.loc[idx, :]
        features_idx = features.loc[idx, :]

        tracks_idx = tracks_idx[[('track', 'genre_top')]]
        labels = np.array(tracks_idx).ravel()
        enc = LabelEncoder()
        labels = torch.tensor(enc.fit_transform(labels))

        feats = np.array(features_idx)
        scaler = StandardScaler(copy=False)
        feats = torch.tensor(scaler.fit_transform(feats), dtype=torch.float32)

        datasets.append(MusicalFeaturesDataset(
            features=feats,
            genres=labels
        ))

    return datasets


# In[5]:


data_dir = '/cluster/scratch/lmautner/mgr/data' if on_cluster else '../../data'

dataset_train, dataset_test = read_data(data_dir)


# In[6]:


X_train = dataset_train.features
y_train = dataset_train.genres
X_test = dataset_test.features
y_test = dataset_test.genres


# ### Model Definition

# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


svc_rbf = SVC(kernel='rbf') if on_cluster else LogisticRegression()


# In[8]:


dimensions = [10, 50, 100, 200, 400, 518]


# ### Reduction Baselines

# In[9]:


results_dir = '/cluster/scratch/lmautner/mgr/results' if on_cluster else '../../data/results'


def measure_performance(dir_name: str, preparation_method, preprocessing_method):
    scores = []
    times_preprocess = []
    times_classify = []
    times = []

    t_prep = time.time()
    prep_result = preparation_method()
    time_prepare = time.time() - t_prep

    for dim in dimensions:
        svc = sklearn.base.clone(svc_rbf)

        t_start = time.time()

        Xtr, Xte = preprocessing_method(prep_result, dim)

        t_ready = time.time()

        svc.fit(X=Xtr, y=y_train)
        scores.append(svc.score(Xte, y_test))

        t_end = time.time()

        times_preprocess.append(t_ready - t_start)
        times_classify.append(t_end - t_ready)
        times.append(t_end - t_start)

    scores_df = pd.DataFrame(scores, index=dimensions, columns=['score'])
    times_preprocess_df = pd.DataFrame(times_preprocess, index=dimensions, columns=['time pre-process (sec)'])
    times_classify_df = pd.DataFrame(times_classify, index=dimensions, columns=['time classify (sec)'])
    time_prepare_df = pd.DataFrame([time_prepare], index=[0], columns=['time prepare (sec)'])

    times_df = pd.DataFrame(times, index=dimensions, columns=['time (sec)'])
    times_df['time (sec)'] = times_df['time (sec)'] + time_prepare

    path = f'{results_dir}/{dir_name}'
    if not os.path.exists(path):
            os.mkdir(path)

    scores_df.to_csv(f'{path}/scores.csv')
    times_preprocess_df.to_csv(f'{path}/times_preprocess.csv')
    times_classify_df.to_csv(f'{path}/times_classify.csv')
    time_prepare_df.to_csv(f'{path}/time_prepare.csv')
    times_df.to_csv(f'{path}/times.csv')


# #### SVD

# In[10]:


from sklearn.decomposition import TruncatedSVD


def prepare_svd():
    return None


def preprocess_svd(prep_res, dim):
    svd = TruncatedSVD(n_components=dim)
    svd.fit(X_train)
    return svd.transform(X_train), svd.transform(X_test)


measure_performance(dir_name='svd', preparation_method=prepare_svd, preprocessing_method=preprocess_svd)


# #### Random Forest Importance

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


def prepare_random_forest():
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X_train, y_train)

    important_feature_indices = np.argsort(forest.feature_importances_)[::-1]
    return important_feature_indices


def preprocess_random_forest(prep_res, dim):
    important_feature_indices = prep_res.copy()[:dim]  # taking the first dim of the importance-sorted features
    return X_train[:, important_feature_indices], X_test[:, important_feature_indices]


measure_performance(dir_name='random_forest', preparation_method=prepare_random_forest, preprocessing_method=preprocess_random_forest)


# ### Single-Feature Predictiveness

# In[ ]:


cutoff_size_train = X_train.shape[0] if on_cluster else 100
cutoff_size_test = X_test.shape[0] if on_cluster else 100

def prepare_single_feature():
    single_feature_scores = []
    for fi in range(X_train.shape[1]):
        svc_single = sklearn.base.clone(svc_rbf)

        f_train = X_train[:cutoff_size_train, [fi]]
        f_test = X_test[:cutoff_size_test, [fi]]

        svc_single.fit(f_train, y_train[:cutoff_size_train])
        single_feature_scores.append(svc_single.score(f_test, y_test[:cutoff_size_test]))

    return np.argsort(np.array(single_feature_scores))[::-1]


def preprocess_single_feature(prep_res, dim):
    important_feature_indices = prep_res.copy()[:dim]  # taking the first dim of the importance-sorted single features
    return X_train[:, important_feature_indices], X_test[:, important_feature_indices]


measure_performance(dir_name='single_feature', preparation_method=prepare_single_feature, preprocessing_method=preprocess_single_feature)

