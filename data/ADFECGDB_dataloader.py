import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import pyedflib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, lfilter
# import wfdb
from options.ADFECG_parameter import parser

class ADFECGDB_Dataset(Dataset):

    def __init__(self, root_dirA, fileNames, postfix_set, overlap):

        self.overlap = overlap  # range between [0,1)
        self.postfix_set = postfix_set 
        if postfix_set == "edf":
            self.root_dirA = root_dirA
            self.data_type = "edf"
            self.fileNames = fileNames 
            self.edf_files = [f for f in os.listdir(root_dirA) if f.endswith(".edf")]  
            self.num_files = len(self.edf_files)  
            AECG_list = []
            FECG_list = []
            for i in range(5):
                AECG, FECG = self.read_edf_signal(sigNum=i, path=root_dirA)
                AECG_sample = self.WindowingSig(AECG[0, :])
                FECG_sample = self.WindowingSig(FECG[0, :])
                AECG_list.append(AECG_sample)
                FECG_list.append(FECG_sample)
            self.AECG_samples = torch.cat(AECG_list, dim=0)
            self.FECG_samples = torch.cat(FECG_list, dim=0)

            print("Dataset loaded from {} ...".format(
                root_dirA))  

    def read_edf_signal(self, sigNum, path="./dataset/ADFECGDB/"):

        file_name = path +'/'+ self.fileNames[sigNum] 
        f = pyedflib.EdfReader(file_name)  
        n = f.signals_in_file 

        AECG = np.zeros((n - 1, f.getNSamples()[0]))  
        FECG = np.zeros((1, f.getNSamples()[0])) 
        FECG[0, :] = f.readSignal(0) 
        FECG_filtered = self.butter_bandpass_filter(FECG, 2, 100, 1000)
        FECG_filtered = self.notch_filter(FECG_filtered, 1000.0, 50.0, 25.0)

        copied_FECG_filtered = FECG_filtered.copy()
        tensor_FECG_filtered = torch.from_numpy(copied_FECG_filtered).float()
        FECG = tensor_FECG_filtered

        for i in np.arange(1, n):
            AECG[i - 1, :] = f.readSignal(i) 
        AECG_filtered = self.butter_bandpass_filter(AECG, 2, 100, 1000)
        AECG_filtered = self.notch_filter(AECG_filtered, 1000.0, 50.0, 25.0)

        copied_AECG_filtered = AECG_filtered.copy()
        tensor_AECG_filtered = torch.from_numpy(copied_AECG_filtered).float()
        AECG = tensor_AECG_filtered

        # resampling to 200Hz
        AECG = signal.resample(AECG, int(AECG.shape[1] / 5), axis=1) 
        FECG = signal.resample(FECG, int(FECG.shape[1] / 5), axis=1)  

        AECG = torch.from_numpy(AECG).float()   
        FECG = torch.from_numpy(FECG).float()   
        return AECG, FECG

    def WindowingSig(self, signal, window_size=200):
        hop_size = int(window_size*(1-self.overlap))
        NumSegements = int(np.floor((len(signal) - window_size) / hop_size) + 1)
        segments = np.zeros((NumSegements, int(window_size)), dtype=np.float32)   # the shape of samples: (NumSegements, window_size)
        for i in range(int(NumSegements)):
            segments[i, :] = signal[i * hop_size: i * hop_size + window_size]
            segments[i, :] = segments[i, :] * np.hamming(window_size)
        segments = torch.tensor(segments, dtype=torch.float32).reshape((-1, window_size, 1))
        return segments

    def Z_score(self, signal):
        """
        this function is to realize the normalization, and the amplitude of this Z_scored signal is between [-1, 1]
        """
        signal = signal.reshape(-1, min(np.shape(signal)))
        signal_mean = np.mean(signal, axis=0)
        length = len(signal)

        vari = np.sqrt(np.sum((signal - signal_mean) ** 2, axis=0) / length)
        Z_scored_signal = (signal - signal_mean) / vari
        return Z_scored_signal

    def __getitem__(self, idx):
        if self.data_type == "edf":
            self.A_sample = self.AECG_samples[idx]
            self.B_sample = self.FECG_samples[idx]

        return self.A_sample, self.B_sample

    def __len__(self):
        return min(len(self.AECG_samples), len(self.FECG_samples))

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, axis=1):

        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=axis)
        return y

    def notch_filter(self, signal, fs, notch_freq=50.0, quality_factor=30.0):
        """
        this function is to design a notch filter to remove powerline interference(50Hz)
        --------------------------------------------------------------------------------
        Args:
            signal: Input signal  ndarray
            fs: the sampling rate of the signal  float
            notch_freq: Frequency to be removed (in Hz)
            quality_factor: Quality factor of the notch filter.
        --------------------------------------------------------------------------------
        Returns:
            filtered_signal: the filtered signal   ndarray
        """
        # design the notch filter
        b, a = iirnotch(notch_freq, quality_factor, fs)

        # apply the notch filter
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal

def get_ADFECGDB_dataloader(root_dirA="./dataset/ADFECGDB", batch_size=8, overlap=0):
    postfix_set = "edf"
    fileNames = os.listdir(root_dirA)
    dataset = ADFECGDB_Dataset(root_dirA=root_dirA, fileNames=fileNames, postfix_set=postfix_set, overlap=overlap)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    args = parser.parse_args()
    train_dataloader, test_dataloader = get_ADFECGDB_dataloader(root_dirA="../dataset/ADFECGDB", batch_size=args.batch_size)
    for idx, batch in enumerate(train_dataloader):
        if idx > 5:
            break
        print(batch[0].size(), batch[1].size())