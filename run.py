# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import warnings
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, BaggingClassifier
from sklearn.tree import ExtraTreeRegressor, ExtraTreeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, pipeline


import xgboost as xgb
import lightgbm as lgb
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Add, Dense, Flatten, BatchNormalization, Activation, UpSampling2D, Embedding
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import losses, metrics, optimizers
from keras.applications import MobileNet
# from keras.applications.mobilenet import relu6, DepthwiseConv2D

import tensorflow as tf
import keras

from glob import glob

import simplejson as json

import pickle
import gensim
import jieba
import itertools

from keras.models import Sequential
from keras.layers import Dense, Dropout, Concatenate, Add, SpatialDropout1D
from keras.layers import Embedding
# from keras.layers import LSTM, TimeDistributed, Bidirectional, CuDNNLSTM, CuDNNGRU
from keras.layers import LSTM, TimeDistributed, Bidirectional

import keras.backend as K


LABELS = ['Normal',
'Disease',
'Reason',
'Symptom',
'Test',
'Test_Value',
'Drug',
'Frequency',
'Amount',
'Method',
'Treatment',
'Operation',
'Anatomy',
'Level',
'Duration',
'SideEff',
]


LABELS_0 = ['Normal', 'Test', 'Symptom', 'Treatment', 'Drug', 'Anatomy', ]
LABELS_1 = ['Normal', 'Frequency', 'Duration', 'Amount', 'Method', 'SideEff', ]

LABELS_DICT = dict(zip(LABELS, range(len(LABELS))))

LABELS_DICT_IDX = {str(v):k for k, v in LABELS_DICT.items()}

ARG_0a_SET = set(['Test', 'Symptom', 'Treatment', 'Drug', 'Anatomy'])

ARG_1a_SET = set(['Frequency', 'Duration', 'Amount', 'Method', 'SideEff'])

ARG_0b = 'Disease'
ARG_1b = 'Drug'


TRAN_TAB = str.maketrans(u'！？【】『』〔〕（）％＃＠＆１２３４５６７８９０“”’‘', u'!?[][][]()%#@&1234567890""\'\'')

TOTAL_SIZE = None

TRAIN_LENGTH_HALF = 160
TRAIN_LENGTH = (TRAIN_LENGTH_HALF*2)

LEN_LABELS = len(LABELS_DICT)


NOISE_RATIO =  0.02
NOISE_ARGA = 0.01

MARGIN_LIST = [-140, -112, -84, -56, -28, 0, 28, 56, 84, 112, 140]

W2V_LEN_TEXT = 64
W2V_LEN_ARGA = 32
W2V_LEN_ARGB = 8

TOTAL_ARGA = LEN_LABELS + 1
TOTAL_ARGB = 3

FLAG_DROP = True

KFOLD = 10

EPOCHS = 25


def remove_semicolon(t):

    if ';' in (t):

        temp = t.split(';')
        temp_0 = temp[0].split(' ')[:-1]
        temp_1 = temp[-1].split(' ')[-1:]

        return temp_0 + temp_1

    else:
        return t.split(' ')

def split_arg(t):

    t_list = t.split(' ')

    if '_' in t_list[0]:
        arg_a, arg_b = t_list[0].split('_')
    elif '-' in t_list[0]:
        arg_a, arg_b = t_list[0].split('-')

    arg1 = t_list[1].split(':')[1]
    arg2 = t_list[2].split(':')[1]

    return [arg_a, arg_b, arg1, arg2]

def get_df(path, arg=False, filted=False):

    if arg:
        anno = pd.read_csv(path, sep='\t', header=None)

        anno = anno[anno[0].str.startswith('R')].reset_index(drop=True)

        anno_1 = np.vstack((anno[1].apply(split_arg)))

        anno_df = pd.DataFrame()
        anno_df['id'] = anno[0]
        anno_df['arg_a'] = anno_1[:, 0]
        anno_df['arg_b'] = anno_1[:, 1]
        anno_df['arg_1'] = anno_1[:, 2]
        anno_df['arg_2'] = anno_1[:, 3]

        return anno_df, anno

    else:
        anno = pd.read_csv(path, sep='\t', header=None)

        anno = anno[anno[0].str.startswith('T')].reset_index(drop=True)

        anno_1 = np.vstack((anno[1].apply(remove_semicolon)))

        anno_df = pd.DataFrame()
        anno_df['id'] = anno[0]
        anno_df['label'] = anno_1[:, 0]
        anno_df['idx_0'] = anno_1[:, 1].astype(int)
        anno_df['idx_1'] = anno_1[:, 2].astype(int)
        anno_df['text'] = anno[2]

        if filted:
            anno_df = anno_df[anno_df['label'].isin(ARG_ALL)].reset_index(drop=True)

    return anno_df, anno

def text_filter(text):

    _l = len(text)
    _text = text.translate(TRAN_TAB)
    assert _l == len(_text)

    return _text

def _add_anno_to_arg(df_anno, df_arg):

    anno_a = df_anno # train_anno_dict.get('0')
    anno_b = df_arg # train_arg_dict.get('0')

    arg1_df = (pd.merge(anno_b[['id', 'arg_1', 'arg_a']],
              anno_a[['id', 'idx_0', 'idx_1', 'label', 'text']],
              left_on=['arg_1'],
              right_on=['id']).drop(columns=['arg_1', 'id_y'])).rename(columns={'id_x':'id',
                                                                                'idx_0':'arg_1_idx_0',
                                                                                'idx_1':'arg_1_idx_1',
                                                                                'label':'arg_a_label',
                                                                                'text':'arg_a_text'})
    arg2_df = (pd.merge(anno_b[['id', 'arg_2', 'arg_b']],
              anno_a[['id', 'idx_0', 'idx_1', 'label', 'text']],
              left_on=['arg_2'],
              right_on=['id']).drop(columns=['arg_2', 'id_y'])).rename(columns={'id_x':'id',
                                                                                'idx_0':'arg_2_idx_0',
                                                                                'idx_1':'arg_2_idx_1',
                                                                                'label':'arg_b_label',
                                                                                'text':'arg_b_text'})
    return pd.merge(pd.merge(anno_b, arg1_df), arg2_df)

def add_anno_to_arg(df_anno_dict, df_arg_dict):

    arg_dict = {}

    for k in df_anno_dict:
        anno = _add_anno_to_arg(df_anno_dict.get(k), df_arg_dict.get(k))
        idx_all = np.sort(anno[['arg_1_idx_0', 'arg_1_idx_1', 'arg_2_idx_0', 'arg_2_idx_1']].values, axis=1)
        idx_len = idx_all[:, -1] - idx_all[:, 0]
        anno['idx_len_max'] = idx_len
        anno['idx_len_mid'] = (anno['arg_2_idx_1']+anno['arg_2_idx_0'])/2 - (anno['arg_1_idx_1']+anno['arg_1_idx_0'])/2
        anno['arg_len'] = anno['arg_2'].str[1:].astype(int) - anno['arg_1'].str[1:].astype(int)
        anno['len_abs_min'] = anno.apply(lambda x: np.min((np.abs(x['idx_len_max']), np.abs(x['idx_len_mid']), np.abs(x['arg_len']))), axis=1)

#         anno = anno[~(anno['arg_a'] != anno['arg_a_label']) | (anno['arg_b'] != anno['arg_b_label'])].reset_index(drop=True)
        anno = anno[(anno['arg_a'] == anno['arg_a_label']) & (anno['arg_b'] == anno['arg_b_label'])].reset_index(drop=True)

#         thres = np.mean(anno['idx_len_max']) + np.std(anno['idx_len_max']) * 2
#         anno = anno[(anno['idx_len_max'] < thres) | (anno['idx_len_max'] < 350)]
#         anno = anno[(anno['idx_len_max'] < thres)]

        arg_dict[k] = anno

    return arg_dict


def get_a_labels_dict(anno_dict, text_dict, labels_dict):
    train_ab_lables_dict = {}
    for k, v in anno_dict.items():

        _labels = np.ones((len(text_dict.get(k)), 1), dtype=np.uint8) * labels_dict.get('Normal')

        for idx, row in enumerate(v.itertuples()):

            _label = row[2]

            if _label in labels_dict:

                _idx_0 = int(row[3])
                _idx_1 = int(row[4])

                if _idx_0 >= _idx_1:
                    print('Bad idx', k, row)
                    break

                _labels[_idx_0:_idx_1] = labels_dict.get(_label)

        train_ab_lables_dict[k] = _labels

    return train_ab_lables_dict

def get_aa_labels_dict(anno_dict, text_dict, labels_set):
    train_ab_lables_dict = {}
    for k, v in anno_dict.items():

        _labels = np.zeros((len(text_dict.get(k)), 1), dtype=np.uint8)

        for idx, row in enumerate(v.itertuples()):

            _label = row[2]

            if _label in labels_set:

                _idx_0 = int(row[3])
                _idx_1 = int(row[4])

                if _idx_0 >= _idx_1:
                    print('Bad idx', k, row)
                    break

#                 _labels[_idx_0:_idx_1] = labels_dict.get(_label)
                _labels[_idx_0:_idx_1] = 1

        train_ab_lables_dict[k] = _labels

    return train_ab_lables_dict

def _get_set_b_list_dict(b_dict, set_b, arg_df, len_text):

    for _b in set_b:
        _df = arg_df[arg_df['arg_2'] == _b]
        _x = np.zeros((len_text), dtype=np.uint8)

        for idx_b, row_b in enumerate(_df.itertuples()):

            assert row_b[6] >= 0
            assert row_b[7] >= 0
            assert row_b[6] <= len_text
            assert row_b[7] <= len_text

            _x[row_b[6]:row_b[7]] = 1

        b_dict[_b] = _x

    return b_dict

def _get_meta_data(data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list,
                   text_encoded, b_dict, list_a, df_b, x_idx, len_text, set_b, list_aa):

    np.random.seed(522)

    for idx_b, row_b in enumerate(df_b.itertuples()):

        _, id_b, label_b, idx_b_0, idx_b_1, _ = row_b

        _idx_b_mid = int((idx_b_0 + idx_b_1)/2)

        for margin in MARGIN_LIST:

            idx_b_mid = _idx_b_mid + margin

            _left = idx_b_mid - TRAIN_LENGTH_HALF
            _right = idx_b_mid + TRAIN_LENGTH_HALF

            idx_left = max(0, _left)
            idx_right = min(len_text, _right)

            idx_pad_left = np.abs(min(0, _left))
            idx_pad_right = np.abs(min(0, len_text - _right))

            # x arg b
            _b = np.zeros((TRAIN_LENGTH, 1), dtype=np.uint8)
            b_start = idx_pad_left + (idx_b_0 - idx_left)
            b_end = idx_pad_left + (idx_b_1 - idx_left)

            if (b_start < 0) or (b_end < 0) or (b_start > TRAIN_LENGTH) or (b_end > TRAIN_LENGTH):
                continue

            assert b_start >= 0
            assert b_end >= 0
            assert b_start <= TRAIN_LENGTH
            assert b_end <= TRAIN_LENGTH

            _b[b_start:b_end] = x_idx+1

            # x arg a
            _a = np.pad(list_a[idx_left:idx_right], ((idx_pad_left, idx_pad_right), (0, 0)), mode='constant', constant_values=0).copy()

            # x arg aa
            _aa = np.pad(list_aa[idx_left:idx_right], ((idx_pad_left, idx_pad_right), (0, 0)), mode='constant', constant_values=0).copy()

            # x text
            _x = np.pad(text_encoded[idx_left:idx_right], (idx_pad_left, idx_pad_right), mode='constant', constant_values=0).copy()

            mask_t = None
            mask_a = None
            if NOISE_RATIO != 0:
                mask = np.argwhere(np.squeeze(_a + _b) == 0).reshape(-1)[idx_pad_left:TRAIN_LENGTH-idx_pad_right]
#                 mask = np.argwhere(np.squeeze(_b) == 0).reshape(-1)[idx_pad_left:TRAIN_LENGTH-idx_pad_right]
                if len(mask)>0:
                    mask_t = np.random.choice(mask, int(len(mask)*NOISE_RATIO))
                    _x[mask_t] = 0

            if NOISE_ARGA != 0:
                mask = np.argwhere(np.squeeze(_a) != 0).reshape(-1) #[idx_pad_left:TRAIN_LENGTH-idx_pad_right]
                if len(mask)>0:
#                     mask_a = np.random.choice(mask, int(np.ceil(len(mask)*NOISE_ARGA)))
                    mask_a = np.random.choice(mask, int(len(mask)*NOISE_ARGA))
                    _x[mask_a] = 0
#                     _a[mask_a] = 0
#                     _aa[mask_a] = 0

            data_x_arga_list.append(_a)
            data_x_argaa_list.append(_aa)
            data_x_argb_list.append(_b)
            data_x_text_list.append(_x)

            # y
            _y = None
            if id_b in set_b:
                _y = np.pad(
                        b_dict.get(id_b)[idx_left:idx_right],
                        (idx_pad_left, idx_pad_right),
                        mode='constant', constant_values=0)[..., np.newaxis].copy()
#                 if mask_a is not None:
#                     _y[mask_a] = 0

            else:
                _y = np.zeros((TRAIN_LENGTH, 1), dtype=np.uint8)

            data_y_list.append(_y)


    return (data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list)

def get_meta_data(data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list,
                  text_encoded, anno_df, list_a, arg_df, list_0a, list_1a):

    len_text = len(text_encoded)

    df_0b = anno_df[anno_df['label'] == ARG_0b].sort_values('idx_0').reset_index(drop=True)
    df_1b = anno_df[anno_df['label'] == ARG_1b].sort_values('idx_0').reset_index(drop=True)

    set_0b = None
    set_1b = None
    b_dict = None
    if arg_df is not None:
        set_0b = arg_df[arg_df['arg_b'] == ARG_0b]['arg_2'].unique()
        set_1b = arg_df[arg_df['arg_b'] == ARG_1b]['arg_2'].unique()

        b_dict = {}
        b_dict = _get_set_b_list_dict(b_dict, set_0b, arg_df, len_text)
        b_dict = _get_set_b_list_dict(b_dict, set_1b, arg_df, len_text)

#     print(df_0b.shape, df_1b.shape, list_0a.shape, list_1a.shape, len(set_0b), len(set_1b))

#     list_a = list_0a
    list_aa = list_0a
    df_b = df_0b
    set_b = set_0b
    x_idx = 0

    (data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list) = \
        _get_meta_data(data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list,
                       text_encoded, b_dict, list_a, df_b, x_idx, len_text, set_b, list_aa)

#     list_a = list_1a
    list_aa = list_1a
    df_b = df_1b
    set_b = set_1b
    x_idx = 1

    (data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list) = \
        _get_meta_data(data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list,
                       text_encoded, b_dict, list_a, df_b, x_idx, len_text, set_b, list_aa)

    return (data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list)

from keras.layers import LSTM, GRU


emb_weights = None

def get_model(emb_weights=emb_weights):

    inputs_text = Input((TRAIN_LENGTH, ))
    emb_text = Embedding(input_dim=TOTAL_SIZE+1,output_dim=W2V_LEN_TEXT, weights=[emb_weights], trainable=False)(inputs_text)
    emb_text = SpatialDropout1D(0.3)(emb_text)
    # x_text = Bidirectional(CuDNNLSTM(128, return_sequences=True))(emb_text)
    x_text = Bidirectional(LSTM(128, return_sequences=True))(emb_text)
    x_text = Dropout(0.5)(x_text)

    inputs_arg_a = Input((TRAIN_LENGTH, ))
    emb_arga = Embedding(input_dim=TOTAL_ARGA,output_dim=W2V_LEN_ARGA)(inputs_arg_a)
    emb_arga = SpatialDropout1D(0.3)(emb_arga)
    # x_arg_a = Bidirectional(CuDNNLSTM(128, return_sequences=True))(emb_arga)
    x_arg_a = Bidirectional(LSTM(128, return_sequences=True))(emb_arga)
    x_arg_a = Dropout(0.5)(x_arg_a)

    inputs_arg_b = Input((TRAIN_LENGTH, ))
    emb_argb = Embedding(input_dim=TOTAL_ARGB,output_dim=W2V_LEN_ARGB)(inputs_arg_b)
    # x_arg_b = Bidirectional(CuDNNLSTM(128, return_sequences=True))(emb_argb)
    x_arg_b = Bidirectional(LSTM(128, return_sequences=True))(emb_argb)

    inputs_all = Concatenate()([x_text, x_arg_a, x_arg_b])

    # x0 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inputs_all)
    x0 = Bidirectional(LSTM(128, return_sequences=True))(inputs_all)
    # x1 = Bidirectional(CuDNNGRU(128, return_sequences=True))(inputs_all)
    x1 = Bidirectional(GRU(128, return_sequences=True))(inputs_all)

    x = Concatenate()([x0, x1])

    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(1))(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=[inputs_text, inputs_arg_a, inputs_arg_b], outputs=x)
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.RMSprop(),
                  metrics=[metrics.binary_accuracy, metrics.binary_crossentropy])

    return model


def _get_test_meta_data(b_list, idx_pad_left_list, idx_left_list, idx_right_list,
                        x_text_list, x_arga_list, x_argaa_list, x_argb_list, y_list, text_encoded, list_a, df_b, x_idx, len_text, list_aa):


    for idx_b, row_b in enumerate(df_b.itertuples()):

        _, id_b, label_b, idx_b_0, idx_b_1, _ = row_b

        _idx_b_mid = int((idx_b_0 + idx_b_1)/2)

        for margin in MARGIN_LIST:

            idx_b_mid = _idx_b_mid + margin

            _left = idx_b_mid - TRAIN_LENGTH_HALF
            _right = idx_b_mid + TRAIN_LENGTH_HALF

            idx_left = max(0, _left)
            idx_right = min(len_text, _right)

            idx_pad_left = np.abs(min(0, _left))
            idx_pad_right = np.abs(min(0, len_text - _right))

            # x arg b
            _b = np.zeros((TRAIN_LENGTH, 1), dtype=np.uint8)
            b_start = idx_pad_left + (idx_b_0 - idx_left)
            b_end = idx_pad_left + (idx_b_1 - idx_left)

            if (b_start < 0) or (b_end < 0) or (b_start > TRAIN_LENGTH) or (b_end > TRAIN_LENGTH):
                continue

            assert b_start >= 0
            assert b_end >= 0
            assert b_start <= TRAIN_LENGTH
            assert b_end <= TRAIN_LENGTH


            _b[b_start:b_end] = x_idx+1

            x_argb_list.append(_b)


            # b_list
            b_list.append(id_b)

            # idx_pad_left
            idx_pad_left_list.append(idx_pad_left)

            # idx_left_list
            idx_left_list.append(idx_left)

            # idx_right_list
            idx_right_list.append(idx_right)

            # x arg a
            _a = np.pad(list_a[idx_left:idx_right], ((idx_pad_left, idx_pad_right), (0, 0)), mode='constant', constant_values=0)

            x_arga_list.append(_a)

            # x arg aa
            _aa = np.pad(list_aa[idx_left:idx_right], ((idx_pad_left, idx_pad_right), (0, 0)), mode='constant', constant_values=0)

            x_argaa_list.append(_aa)

            # x text
            _x = np.pad(text_encoded[idx_left:idx_right], (idx_pad_left, idx_pad_right), mode='constant', constant_values=0).copy()

            x_text_list.append(_x)

    return b_list, idx_pad_left_list, idx_left_list, idx_right_list, x_text_list, x_arga_list, x_argaa_list, x_argb_list, y_list

def get_test_meta_data(x_text_list, x_arga_list, x_argaa_list, x_argb_list, y_list, text_encoded, anno_df, list_a, list_0a, list_1a):

    len_text = len(text_encoded)

    df_0b = anno_df[anno_df['label'] == ARG_0b].sort_values('idx_0').reset_index(drop=True)
    df_1b = anno_df[anno_df['label'] == ARG_1b].sort_values('idx_0').reset_index(drop=True)

    b_list = []
    idx_pad_left_list = []
    idx_left_list = []
    idx_right_list = []

#     list_a = list_0a
    list_aa = list_0a
    df_b = df_0b
    x_idx = 0

    b_list, idx_pad_left_list, idx_left_list, idx_right_list, x_text_list, x_arga_list, x_argaa_list, x_argb_list, y_list = \
        _get_test_meta_data(b_list, idx_pad_left_list, idx_left_list, idx_right_list, x_text_list, x_arga_list, x_argaa_list, x_argb_list, y_list, text_encoded, list_a, df_b, x_idx, len_text, list_aa)

#     list_a = list_1a
    list_aa = list_1a
    df_b = df_1b
    x_idx = 1

    b_list, idx_pad_left_list, idx_left_list, idx_right_list, x_text_list, x_arga_list, x_argaa_list, x_argb_list, y_list = \
        _get_test_meta_data(b_list, idx_pad_left_list, idx_left_list, idx_right_list, x_text_list, x_arga_list, x_argaa_list, x_argb_list, y_list, text_encoded, list_a, df_b, x_idx, len_text, list_aa)

    return (b_list, idx_pad_left_list, idx_left_list, idx_right_list, x_text_list, x_arga_list, x_argaa_list, x_argb_list, y_list)


if __name__ == "__main__":

    # Part1 train
    train_ann_file_list = glob('./Demo/DataSets/ruijin_round2_train/ruijin_round2_train/*.ann')
    train_txt_file_list = glob('./Demo/DataSets/ruijin_round2_train/ruijin_round2_train/*.txt')

    test_txt_file_list = glob('./Demo/DataSets/ruijin_round2_test_b/ruijin_round2_test_b/*.txt')
    test_ann_file_list = glob('./Demo/DataSets/ruijin_round2_test_b/ruijin_round2_test_b/*.ann')

    test_ann_file_list_dict = {str(path.split('/')[-1].split('.')[0]):path for path in test_ann_file_list}

    train_text_dict = {}
    for path in train_txt_file_list:
        with open(path) as f:
            train_text_dict[str(path.split('/')[-1].split('.')[0])] = list(text_filter(str(f.read())))


    train_anno_dict = {str(path.split('/')[-1].split('.')[0]):get_df(path, filted=False)[0] for path in train_ann_file_list}

    train_arg_dict = {str(path.split('/')[-1].split('.')[0]):get_df(path, arg=True)[0] for path in train_ann_file_list}

    train_arg_dict = add_anno_to_arg(train_anno_dict, train_arg_dict)


    train_text_set = set()
    for k, v in train_text_dict.items():
        train_text_set = train_text_set | set(v)


    train_encode_dict = dict(zip(sorted(list(train_text_set)), range(len(train_text_set))))

    TOTAL_SIZE = len(train_text_set)+1

    train_text_encoded_dict = {}
    for k, v in train_text_dict.items():
        train_text_encoded_dict[k] = list(map(lambda x: train_encode_dict.get(x)
                                              if x in train_encode_dict else 0, v))


    train_encode_dict_idx = {str(v):k for k, v in train_encode_dict.items()}

    train_0a_lables_dict = get_aa_labels_dict(train_anno_dict, train_text_dict, LABELS_0)
    train_1a_lables_dict = get_aa_labels_dict(train_anno_dict, train_text_dict, LABELS_1)

    train_a_lables_dict = get_a_labels_dict(train_anno_dict, train_text_dict, LABELS_DICT)


    data_x_text_list = []
    data_x_arga_list = []
    data_x_argaa_list = []
    data_x_argb_list = []
    data_y_list = []


    for k in train_text_encoded_dict:

        t0 = train_text_encoded_dict.get(k)
        anno_0 = train_anno_dict.get(k)
        arg_0 = train_arg_dict.get(k)
        list_a = train_a_lables_dict.get(k)
        list_0a = train_0a_lables_dict.get(k)
        list_1a = train_1a_lables_dict.get(k)

        (data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list) = \
            get_meta_data(data_x_text_list, data_x_arga_list, data_x_argaa_list, data_x_argb_list, data_y_list,
                          text_encoded=t0, anno_df=anno_0, list_a=list_a, arg_df=arg_0, list_0a=list_0a, list_1a=list_1a)


    model_ft = gensim.models.FastText([jieba.lcut(''.join(v)) for v in train_text_dict.values()], size=W2V_LEN_TEXT, word_ngrams=9, window=24, min_count=1, workers=8, seed=522)


    emb_weights = np.zeros((TOTAL_SIZE+1, W2V_LEN_TEXT))

    count_emb_0 = 0

    for k, v in train_encode_dict.items():

        _k = str(k)
        if _k in model_ft.wv:
            emb_weights[v] = model_ft.wv[_k]
        else:
            count_emb_0 += 0


    # data_x_text_list = np.squeeze(data_x_text_list).astype(np.uint16)
    # data_x_arga_list = np.squeeze(data_x_arga_list).astype(np.uint8)
    # data_x_argaa_list = np.asarray(data_x_argaa_list, dtype=np.uint8)
    # data_x_argb_list = np.squeeze(data_x_argb_list).astype(np.uint8)
    # data_y_list = np.asarray(data_y_list, dtype=np.uint8)

    data_x_text_list = np.squeeze(data_x_text_list).astype(np.uint16)[:128]
    data_x_arga_list = np.squeeze(data_x_arga_list).astype(np.uint8)[:128]
    data_x_argaa_list = np.asarray(data_x_argaa_list, dtype=np.uint8)[:128]
    data_x_argb_list = np.squeeze(data_x_argb_list).astype(np.uint8)[:128]
    data_y_list = np.asarray(data_y_list, dtype=np.uint8)[:128]


    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=522).split(data_x_text_list)
    train_idx, valid_idx = next(kf)


    train_x_text_list = data_x_text_list[train_idx]
    train_x_arga_list = data_x_arga_list[train_idx]
    train_x_argaa_list = data_x_argaa_list[train_idx]
    train_x_argb_list = data_x_argb_list[train_idx]
    train_y_list = data_y_list[train_idx]

    valid_x_text_list = data_x_text_list[valid_idx]
    valid_x_arga_list = data_x_arga_list[valid_idx]
    valid_x_argaa_list = data_x_argaa_list[valid_idx]
    valid_x_argb_list = data_x_argb_list[valid_idx]
    valid_y_list = data_y_list[valid_idx]

    MODEL_BASE = 's2_01'
    MODEL_NAME = MODEL_BASE + '_final_v4_0'


    model = get_model(emb_weights)


    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    checkpoint = ModelCheckpoint('./Demo/Models/' + MODEL_BASE + '/' + MODEL_NAME + '_{epoch:02d}.hd5', monitor='val_loss', verbose=1,
                                 save_best_only=False, save_weights_only=True, mode='auto', period=1)
    # lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.6, min_delta=1e-5, patience=2, verbose=1, min_lr = 0.00001)


    hist_0 = model.fit(x=[train_x_text_list, train_x_arga_list, train_x_argb_list], y=train_y_list,
              # batch_size=512,
              batch_size=64,
              epochs=EPOCHS,
              shuffle=True,
              validation_data=([valid_x_text_list, valid_x_arga_list, valid_x_argb_list], valid_y_list),
              # callbacks=[checkpoint, lr_reduce],
              callbacks=[checkpoint],
              verbose=2)

    with open('./Demo/Models/' + MODEL_BASE + '/' + MODEL_NAME + '_hist.pkl', 'wb') as f:
        pickle.dump(hist_0.history, f)


    MODEL_BASE = 's2_01'
    MODEL_NAME = MODEL_BASE + '_final_v4_1'


    model = get_model(emb_weights)


    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    checkpoint = ModelCheckpoint('./Demo/Models/' + MODEL_BASE + '/' + MODEL_NAME + '_{epoch:02d}.hd5', monitor='val_loss', verbose=1,
                                 save_best_only=False, save_weights_only=True, mode='auto', period=1)
    # lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.6, min_delta=1e-5, patience=2, verbose=1, min_lr = 0.00001)



    hist_1 = model.fit(x=[train_x_text_list[:, ::-1], train_x_arga_list[:, ::-1], train_x_argb_list[:, ::-1]], y=train_y_list[:, ::-1, :],
              # batch_size=512,
              batch_size=64,
              epochs=EPOCHS,
              shuffle=True,
              validation_data=([valid_x_text_list[:, ::-1], valid_x_arga_list[:, ::-1], valid_x_argb_list[:, ::-1]], valid_y_list[:, ::-1, :]),
              # callbacks=[checkpoint, lr_reduce],
              callbacks=[checkpoint],
              verbose=2)

    with open('./Demo/Models/' + MODEL_BASE + '/' + MODEL_NAME + '_hist.pkl', 'wb') as f:
        pickle.dump(hist_1.history, f)


    # Part2 prediction
    test_text_dict = {}
    for path in test_txt_file_list:
        with open(path) as f:
            test_text_dict[str(path.split('/')[-1].split('.')[0])] = list(text_filter(str(f.read())))



    test_text_encoded_dict = {}
    for k, v in test_text_dict.items():
        test_text_encoded_dict[k] = list(map(lambda x: train_encode_dict.get(x)
                                              if x in train_encode_dict else 0, v))


    test_anno_dict = {str(path.split('/')[-1].split('.')[0]):get_df(path, filted=False)[0] for path in test_ann_file_list}


    test_0a_lables_dict = get_aa_labels_dict(test_anno_dict, test_text_dict, LABELS_0)
    test_1a_lables_dict = get_aa_labels_dict(test_anno_dict, test_text_dict, LABELS_1)

    test_a_lables_dict = get_a_labels_dict(test_anno_dict, test_text_dict, LABELS_DICT)


    model = get_model(emb_weights)

    # PRE_THRES = 0.6
    PRE_THRES = 0.1

    pre_final_dict = {}

    for file_k in test_text_encoded_dict:

        print('Start ', file_k, '---------------------------------')

        # 1. get model data inputs
        print('Part 1...')
        test_x_text_list = []
        test_x_arga_list = []
        test_x_argaa_list = []
        test_x_argb_list = []
        test_y_list = []

        test_b_list = []
        test_idx_pad_left_list = []
        test_idx_left_list = []
        test_idx_right_list = []

        t0 = test_text_encoded_dict.get(file_k)
        anno_0 = test_anno_dict.get(file_k)
        # arg_0 = train_arg_dict.get(file_k)
        arg_0 = None
        list_a = test_a_lables_dict.get(file_k)
        list_0a = test_0a_lables_dict.get(file_k)
        list_1a = test_1a_lables_dict.get(file_k)

        test_b_list, test_idx_pad_left_list, test_idx_left_list, test_idx_right_list, test_x_text_list, test_x_arga_list, test_x_argaa_list, test_x_argb_list, test_y_list = \
            get_test_meta_data(test_x_text_list, test_x_arga_list, test_x_argaa_list, test_x_argb_list, test_y_list,
                          text_encoded=t0, anno_df=anno_0, list_a=list_a, list_0a=list_0a, list_1a=list_1a)

        print(np.shape(test_b_list), np.shape(test_idx_left_list), np.shape(test_x_text_list), np.shape(test_x_arga_list), np.shape(test_x_argaa_list),
              np.shape(test_x_argb_list), np.shape(test_y_list))

        # 2. get predictions
        print('Part 2...')
        test_x_text_list = np.squeeze(test_x_text_list).astype(int)
        test_x_arga_list = np.squeeze(test_x_arga_list).astype(int)
        test_x_argaa_list = np.asarray(test_x_argaa_list, dtype=int)
        test_x_argb_list = np.squeeze(test_x_argb_list).astype(int)

        pre_list = []

        MODEL_BASE = 's2_01'
        MODEL_NAME = MODEL_BASE + '_final_v4_0'
        # for i in [17, 19, 21, 23, 25]:
        for i in [17-1, 19-1, 21-1, 23-1, 25-1]:
            w_p = './Demo/Models/' + MODEL_BASE + '/' + MODEL_NAME + '_%02d.hd5' % (i)
            model.load_weights(w_p)
            # pre_list.append(model.predict([test_x_text_list, test_x_arga_list, test_x_argb_list]))
            pre_list.append(model.predict([test_x_text_list, test_x_arga_list, test_x_argb_list], batch_size=32))

        MODEL_NAME = MODEL_BASE + '_final_v4_1'
        # for i in [17, 19, 21, 23, 25]:
        for i in [17-1, 19-1, 21-1, 23-1, 25-1]:
            w_p = './Demo/Models/' + MODEL_BASE + '/' + MODEL_NAME + '_%02d.hd5' % (i)
            model.load_weights(w_p)
            # pre_list.append(model.predict([test_x_text_list[:, ::-1], test_x_arga_list[:, ::-1], test_x_argb_list[:, ::-1]])[:, ::-1, :])
            pre_list.append(model.predict([test_x_text_list[:, ::-1], test_x_arga_list[:, ::-1], test_x_argb_list[:, ::-1]], batch_size=32)[:, ::-1, :])

        pre = np.mean(pre_list, axis=0)
        print(pre.shape)

        # 3. get pre dict for each output
        print('Part 3...')
        pre_test_dict = {}
        for idx, p in enumerate(pre):

        #     if not len(np.argwhere(p>0.5)) > 0: continue

            _test_b = test_b_list[idx]
            _test_idx_pad_left = test_idx_pad_left_list[idx]
            _test_idx_left = test_idx_left_list[idx]
            _test_idx_right = test_idx_right_list[idx]

            _test_b_label = anno_0[anno_0['id'] == _test_b].iloc[0]['label']

            if _test_b_label == ARG_0b:
    #             _anno_0 = anno_0[(anno_0['label'].isin(LABELS_0_DICT.keys())) & (anno_0['idx_0'] >= _test_idx_left) & (anno_0['idx_1'] <= _test_idx_right)]
                _anno_0 = anno_0[(anno_0['label'].isin(ARG_0a_SET)) & (anno_0['idx_0'] >= _test_idx_left) & (anno_0['idx_1'] <= _test_idx_right)]

            if _test_b_label == ARG_1b:
    #             _anno_0 = anno_0[(anno_0['label'].isin(LABELS_1_DICT.keys())) & (anno_0['idx_0'] >= _test_idx_left) & (anno_0['idx_1'] <= _test_idx_right)]
                _anno_0 = anno_0[(anno_0['label'].isin(ARG_1a_SET)) & (anno_0['idx_0'] >= _test_idx_left) & (anno_0['idx_1'] <= _test_idx_right)]

            if len(_anno_0) > 0:

                for _idx_a, _row_a in enumerate(_anno_0.itertuples()):
                    _, _id_a, _label_a, _idx_a_0, _idx_a_1, _ = _row_a

                    _k = _label_a + _test_b_label + _id_a + _test_b

                    if _k not in pre_test_dict:
                        pre_test_dict[_k] = {}
                        pre_test_dict[_k]['arg_a'] = _label_a
                        pre_test_dict[_k]['arg_b'] = _test_b_label
                        pre_test_dict[_k]['arg_1'] = _id_a
                        pre_test_dict[_k]['arg_2'] = _test_b
                        pre_test_dict[_k]['values'] = 0
                        pre_test_dict[_k]['counts'] = 0

                    _v_l = p[_test_idx_pad_left + _idx_a_0 - _test_idx_left: _test_idx_pad_left + _idx_a_1 - _test_idx_left]
                    assert len(_v_l) > 0
    #                 if len(_v_l) == 0:
    #                     print(_row_a)
    #                     break
                    _value = np.mean(_v_l)
                    pre_test_dict[_k]['values'] = pre_test_dict[_k]['values'] + _value
                    pre_test_dict[_k]['counts'] = pre_test_dict[_k]['counts'] + 1

        print(len(pre_test_dict))

        # 4. filter pre with threshold
        print('Part 4...')
        len_max = 0
        pre_test_df = []
        r_id_idx = 1
        for k, v in pre_test_dict.items():
            _mean = v['values'] / v['counts']

            len_max = len_max if len_max > v['counts'] else v['counts']
            assert len_max <= len(MARGIN_LIST)

            if _mean > PRE_THRES:
                _row = ['R'+str(r_id_idx), v['arg_a']+'_'+v['arg_b']+' '+'Arg1:'+v['arg_1']+' '+'Arg2:'+v['arg_2']]
                pre_test_df.append(_row)
                r_id_idx += 1

        print(len(pre_test_df))

        pre_final_dict[file_k] = pre_test_df

        print('End ', file_k, '---------------------------------')


    MODEL_NAME = 's2_01_final_v4_b'


    for file_k in pre_final_dict:
        _ann_result = pd.read_csv(test_ann_file_list_dict.get(file_k), sep='\t', header=None, names=['id', 'p1', 'p2'])

        _ann_pre_final = np.asarray(pre_final_dict.get(file_k))
        _ann_pre_final_df = pd.DataFrame(columns=['id', 'p1', 'p2'])
        _ann_pre_final_df['id'] = _ann_pre_final[:, 0]
        _ann_pre_final_df['p1'] = _ann_pre_final[:, 1]

        _final_df = _ann_result.append(_ann_pre_final_df, ignore_index=True)



    for file_k in pre_final_dict:
        _ann_result = pd.read_csv(test_ann_file_list_dict.get(file_k), sep='\t', header=None, names=['id', 'p1', 'p2'])

        _ann_pre_final = np.asarray(pre_final_dict.get(file_k))
        _ann_pre_final_df = pd.DataFrame(columns=['id', 'p1', 'p2'])
        _ann_pre_final_df['id'] = _ann_pre_final[:, 0]
        _ann_pre_final_df['p1'] = _ann_pre_final[:, 1]
        _ann_pre_final_df['p1'] = _ann_pre_final_df['p1'].str.replace('SideEff_Drug', 'SideEff-Drug')

        _final_df = _ann_result.append(_ann_pre_final_df, ignore_index=True)

        _final_df.to_csv('./Demo/result/' + MODEL_NAME + '_raw/' + str(file_k) + '.ann', index=False, header=None, sep='\t')








