import os
import copy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from data.data_provider import data_pretreat
from src.utils.timefeatures import time_features
import warnings
import torch

warnings.filterwarnings('ignore')

def corr_with(trends, keys):
    # Correlate by a specific period
    trends_ = copy.deepcopy(trends)
    trends_selected = trends_[keys]
    try:
        correlation_score = trends_.corrwith(trends_selected, axis=0, numeric_only=True)
    except:
        correlation_score = trends_.corrwith(trends_selected, axis=0)
    corr_filtered = pd.DataFrame(correlation_score).fillna(0)
    corr_filtered = corr_filtered.reset_index()
    
    sorted_correlation = correlation_score.sort_values(ascending=True)
    return corr_filtered, sorted_correlation

class Dataset_SHEERM(Dataset):
    
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='SHEERM.csv',
                 target='Net Load', scale=True, timeenc=0, freq='15min', 
                 seasonal_patterns=None, pretreatment=False, cycle=168):
        if size is None:
            self.seq_len = 24 * 4  # 1 day of 15-min data
            self.label_len = 12 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.cycle = cycle  # weekly cycle for 15-min data
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.pretreatment = pretreatment
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('Timestamp')
        df_raw = df_raw[['Timestamp'] + cols + [self.target]]
        #变换顺序
        if self.pretreatment == 1:
           df_raw = data_pretreat.reorder_features(df_raw)
      
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Always use multivariate inputs (all numeric columns except Timestamp); target is the last column.
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['Timestamp']][border1:border2]
        df_stamp['Timestamp'] = pd.to_datetime(df_stamp.Timestamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Timestamp.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.Timestamp.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['Timestamp'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['Timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end],dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark,cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_PSHE(Dataset):
    
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='PSHE.csv',
                 target='NetLoad', scale=True, timeenc=0, freq='30min',
                 seasonal_patterns=None, pretreatment=False, cycle=168):
        if size is None:
            self.seq_len = 24 * 4  # 1 day of 15-min data
            self.label_len = 12 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.cycle = cycle  # weekly cycle for 15-min data
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('timestamp')
        df_raw = df_raw[['timestamp'] + cols + [self.target]]
      
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Always use multivariate inputs (all numeric columns except Timestamp); target is the last column.
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.timestamp.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end],dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark,cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_CityLearn(Dataset):
    
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='SHEERM.csv',
                 target='Net Load', scale=True, timeenc=0, freq='15min', 
                 seasonal_patterns=None, pretreatment=False, cycle=168):
        if size is None:
            self.seq_len = 24 * 4  # 1 day of 15-min data
            self.label_len = 12 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.cycle = cycle  # weekly cycle for 15-min data
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.pretreatment = pretreatment
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('Timestamp')
        df_raw = df_raw[['Timestamp'] + cols + [self.target]]
        #变换顺序
        if self.pretreatment == 1:
           df_raw = data_pretreat.reorder_features(df_raw)
      
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Always use multivariate inputs (all numeric columns except Timestamp); target is the last column.
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['Timestamp']][border1:border2]
        df_stamp['Timestamp'] = pd.to_datetime(df_stamp.Timestamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Timestamp.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.Timestamp.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['Timestamp'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['Timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end],dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark,cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Estonian(Dataset):
    
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='Estonian.csv',
                 target='NetLoad', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, pretreatment=False, cycle=168):
        if size is None:
            self.seq_len = 24  # 1 day of hourly data
            self.label_len = 12
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.cycle = cycle
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.pretreatment = pretreatment
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('Time')
        df_raw = df_raw[['Time'] + cols + [self.target]]
        if self.pretreatment == 1:
            df_raw = data_pretreat.reorder_features(df_raw)
      
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['Time']][border1:border2]
        df_stamp['Time'] = pd.to_datetime(df_stamp.Time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Time.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.Time.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['Time'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['Time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end],dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark,cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
