from os import error
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import math
from torch.utils.data import Dataset
from sklearn.utils import shuffle

class textDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.label_encoder = LabelEncoder()
        self.data_type = data_type
        # self.fac_data, self.fac_label, self.cf_data, self.cf_label, self.field_dims = self.load_fm_dataset(data_path, data_type)
        if data_type == 'train':
            self.data, self.labels, self.field_dims = self.load_fm_dataset(data_path, data_type)
        elif data_type == 'valid':
            self.data, self.labels = self.load_fm_dataset(data_path, data_type)
        elif data_type == 'test':
            self.data, self.labels = self.load_fm_dataset(data_path, data_type)
        else:
            raise NotImplementedError
    def __getitem__(self, index):
            # return self.data[index], self.labels[index]
            return self.data[index], self.labels[index]
    def __len__(self):
        return len(self.data)
    def load_fm_dataset(self, data_path, data_type):
        if data_type == 'train':
        # fac_data = pd.read_csv("/data/MT_wenan_hotel_v2/wenan_data-323168834-1640762850276.txt", sep="\s+", header=0, engine='python', error_bad_lines=False, nrows=7000000)
            print('Reading train Datasets...')
            if data_path == 'beijing':
                fac_data = pd.read_csv('./beijing/train_fac_code.csv')
            elif data_path == 'shanghai':
                fac_data = pd.read_csv('./shanghai/train_fac_code.csv')
            elif data_path == 'new':
                fac_data = pd.read_csv('./data_new/train_fac_code.csv')
            else:
                raise NotImplementedError
            fac_data = fac_data[['value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            fac_data = fac_data.rename(columns={'is_click':'label'})
            # fac_data = fac_data.dropna(axis=0, how = 'any')
            # cf_data_neg = pd.read_csv('./data/counter_neg.csv', nrows=1451151) #13 : 1
            # cf_data_neg = pd.read_csv('./data/counter_neg.csv')
            # cf_data_neg = shuffle(cf_data_neg)
            # cf_data_pos = pd.read_csv('./data/counter_pos.csv') # 111627
            # # cf_data = cf_data_neg.copy()
            # cf_data = pd.concat([cf_data_pos, cf_data_neg.iloc[:1451151]], axis=0)
            # cf_data = shuffle(cf_data)
            if data_path == 'beijing':
                cf_data = pd.read_csv('./beijing/cf_data_code.csv')
            elif data_path == 'shanghai':
                cf_data = pd.read_csv('./shanghai/cf_data_code.csv')
            elif data_path == 'new':
                cf_data = pd.read_csv('./data_new/cf_data_code.csv')
            else:
                raise NotImplementedError
            cf_data = cf_data[['value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            cf_data = cf_data.rename(columns={'is_click':'label'})
            cf_data = cf_data.dropna(axis=0, how='any')
            
            all_data = pd.concat([cf_data, fac_data], axis=0)
            # all_data = fac_data
            all_data = all_data.dropna(axis=0, how='any')
            all_data = all_data.drop_duplicates()
            train_data = all_data.drop("label", axis=1)
            if data_path == 'beijing':
                valid_data = pd.read_csv('./beijing/valid_fac_code.csv')
                test_data = pd.read_csv('./beijing/test_fac_code.csv')
            elif data_path == 'shanghai':
                valid_data = pd.read_csv('./shanghai/valid_fac_code.csv')
                test_data = pd.read_csv('./shanghai/test_fac_code.csv')
            elif data_path == 'new':
                valid_data = pd.read_csv('./data_new/valid_fac_code.csv')
                test_data = pd.read_csv('./data_new/test_fac_code.csv')
            else:
                raise NotImplementedError
            valid_data = valid_data[['value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            test_data = test_data[['value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            valid_data = valid_data.rename(columns={'is_click':'label'})
            test_data = test_data.rename(columns={'is_click':'label'})
            valid_data = valid_data.drop_duplicates()
            test_data = test_data.drop_duplicates()
            valid_feat = valid_data.drop('label', axis=1)
            test_feat = test_data.drop('label', axis=1)
            full = pd.concat([train_data, valid_feat, test_feat], axis=0)
            # for col in train_data.columns:
            #     train_data[col] = train_data[col].astype(str)
            #     train_data[col] = self.label_encoder.fit_transform(train_data[col])
            # fac_dt, fac_label = train_data.iloc[0:fac_data.shape[0]], all_data.label.iloc[0:fac_data.shape[0]]
            # cf_dt, cf_label = train_data.iloc[fac_data.shape[0]:], all_data.label.iloc[fac_data.shape[0]:]
            field_dims = full.nunique()
            return train_data.values, all_data["label"].values, field_dims.values
        elif data_type == 'valid':
            print('Reading valid datasets...')
            if data_path == 'beijing':
                valid_data = pd.read_csv('./beijing/valid_fac_code.csv')
            elif data_path == 'shanghai':
                valid_data = pd.read_csv('./shanghai/valid_fac_code.csv')
            elif data_path == 'new':
                valid_data = pd.read_csv('./data_new/valid_fac_code.csv')
            else:
                raise NotImplementedError
            valid_data = valid_data[[ 'value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            valid_data = valid_data.rename(columns={'is_click':'label'})
            valid_data = valid_data.drop_duplicates()
            feat = valid_data.drop('label', axis=1)
            return feat.values, valid_data["label"].values
        elif data_type == 'test':
            print('Reading test datasets...')
            if data_path == 'beijing':
                test_data = pd.read_csv('./beijing/test_fac_code.csv')
            elif data_path == 'shanghai':
                test_data = pd.read_csv('./shanghai/test_fac_code.csv')
            elif data_path == 'new':
                test_data = pd.read_csv('./data_new/test_fac_code.csv')
            else:
                raise NotImplementedError
            test_data = test_data[[ 'value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            test_data = test_data.rename(columns={'is_click':'label'})
            test_data = test_data.drop_duplicates()
            feat = test_data.drop('label', axis=1)
            return feat.values, test_data["label"].values
        else:
            raise NotImplementedError
        # return fac_dt.values, fac_label.values, cf_dt.values, cf_label.values, field_dims.values
