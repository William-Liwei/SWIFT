from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

from torch.utils.data import DataLoader
from scipy.fft import fft
import numpy as np
import pandas as pd
import os

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True

        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.data == 'm4':
        drop_last = False

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

def measure_periodicity_fft(signal, freq=5):
    p = []
    for i in range(signal.shape[0]):
        fft_output = np.abs(fft(signal[i]))
        magnitude = np.abs(fft_output)**2
        magnitude = magnitude[:freq]
        total_energy = np.sum(magnitude)
        peak_energy = magnitude.max()
        periodicity = peak_energy / total_energy
        p.append(periodicity)
    return np.mean(p)

def period_coeff(args):
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    df_raw = np.array(df_raw.select_dtypes(include=['int', 'float'])).T
    return measure_periodicity_fft(df_raw)
