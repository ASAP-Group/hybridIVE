import os
import torch
from torch.utils.data import Dataset
import numpy as np
from rfc_utils import load_file_list, load_mat_file
import random
from rfc_utils import load_mat_file

# --- code that is same for all datasets ---
class BaseHSEDataset(Dataset):
    def __init__(self, dataset_dir, rellist_path, config, mode=None): #Loading whole dataset into memory - no lazyloading
        self.files_path = load_file_list(rellist_path)
        self.file_num = len(self.files_path)
        self.dataset_dir = dataset_dir
        self.mode = mode
        
        # Load parameters from config
        dataset_config = config.get('dataset', {})
        self.batchsize = dataset_config.get('batchsize', 600)
        self.contextview = dataset_config.get('contextview', 3)
        self.windowlength = dataset_config.get('windowlength', 200)
        self.use_lambda = dataset_config.get('use_lambda', False)
        
        self.add_noise = dataset_config.get('add_noise', False)
        self.snr_range = dataset_config.get('snr_range', [25.0, 30.0])

        # STFT parameters (matching MATLAB generation parameters)
        self.nfft = dataset_config.get('nfft', 512) 
        self.fft_shift = dataset_config.get('fft_shift', 128) 
        self.window_len = dataset_config.get('window_len', self.nfft)
        self.window = torch.hamming_window(self.window_len)  

        self.data_size = self.get_dataset_size()
        print('Loading wav files...')
        self.x = torch.zeros(self.data_size, dtype=torch.complex64)
        self.y = torch.zeros(self.data_size, dtype=torch.complex64)
        if self.use_lambda:
            self.truelambda = torch.zeros(self.file_num, dtype=torch.float32)
        start = 0
        for lpF in range(self.file_num):
            if np.remainder(lpF, 10) == 0:
                print(f'Loading file {lpF} / {self.file_num}')
            file_path = os.path.join(self.dataset_dir, self.files_path[lpF])
            mat = load_mat_file(file_path)
            x_data = torch.from_numpy(mat['x'])[:, 0:self.batchsize, :].detach().to(dtype=torch.complex64)
            y_data = torch.from_numpy(mat['y'])[:, 0:self.batchsize, :].detach().to(dtype=torch.complex64)
            
            # Add noise if enabled (matching MATLAB experimental augmentation)
            if self.add_noise:
                # Generate random SNR in the specified range (25-30 dB as in MATLAB)
                desired_snr = random.uniform(self.snr_range[0], self.snr_range[1])
                x_data = self._add_white_noise(x_data, desired_snr)
            
            self.x[:, start:start+self.batchsize, :] = x_data
            self.y[:, start:start+self.batchsize, :] = y_data
            if self.use_lambda:
                self.truelambda[lpF] = torch.from_numpy(mat['lambdatrue']).detach().to(dtype=torch.float32)
            start += self.batchsize

    def get_dataset_size(self):
        file_path = os.path.join(self.dataset_dir, self.files_path[0])
        mat = load_mat_file(file_path)
        size = [mat['x'].shape[0], self.batchsize * self.file_num, mat['x'].shape[2]]
        return size
    
    def _add_white_noise(self, signal_stft, desired_snr_db):
        """
        Simulate adding white noise in time domain by generating time-domain noise
        and converting to frequency domain using the same STFT parameters as the original data.
        
        Args:
            signal_stft (torch.Tensor): Input STFT signal tensor (complex64)
            desired_snr_db (float): Desired SNR in dB (25-30 dB range as in MATLAB)
            
        Returns:
            torch.Tensor: Signal with added noise, same shape as input
        """

        signal_power = torch.mean(torch.abs(signal_stft)**2)
        snr_linear = 10**(desired_snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate time-domain white noise
        # signal_stft shape: [channels, time_frames, freq_bins]
        num_channels = signal_stft.shape[0]
        num_time_frames = signal_stft.shape[1]
        time_length = (num_time_frames - 1) * self.fft_shift + self.nfft 
        time_noise = torch.randn(num_channels, time_length)
        
        # Apply STFT to get frequency domain noise
        noise_stft = torch.stft(time_noise, 
                               n_fft=self.nfft, 
                               hop_length=self.fft_shift, 
                               window=self.window,
                               return_complex=True)
        
        # torch.stft returns [channels, freq_bins, time_frames] 
        # but we need [channels, time_frames, freq_bins] to match signal_stft
        noise_stft = noise_stft.transpose(1, 2)  # Swap freq and time dimensions
        
        # Ensure dimensions match exactly
        target_shape = signal_stft.shape
        if noise_stft.shape != target_shape:
            # Crop or pad to match exactly
            noise_stft = noise_stft[:target_shape[0], :target_shape[1], :target_shape[2]]
        
        # Scale the noise to have the right power
        current_noise_power = torch.mean(torch.abs(noise_stft)**2)
        if current_noise_power > 0:  # Avoid division by zero
            scale_factor = torch.sqrt(noise_power / current_noise_power)
            noise_stft = noise_stft * scale_factor
        else:
            raise ValueError(f"Noise power was {current_noise_power}. Must be larger than 0.")
        
        # import matplotlib.pyplot as plt 
        # plt.imshow(10*np.log10(np.abs(noise_stft[2,:,:].squeeze())), aspect='auto', origin='lower')
        # plt.xlabel('Time Frame'); plt.ylabel('Frequency Bin'); plt.title('Spectrogram (Channel 3)')
        # plt.colorbar()
        # plt.show()
        # # looks like the same noise from Matlab.

        return signal_stft + noise_stft

    def _get_file_info(self, idx, offset=None):
        """Calculate file information for a given sample index"""
        if self.mode == 'train':
            file_idx = idx
            if offset is None:
                offset = random.randint(0, self.batchsize - self.windowlength - 1)
            return {
                'file_path': self.files_path[file_idx],
                'file_idx': file_idx,
                'sample_idx': idx,
                'frame_offset': offset,
                'mode': self.mode
            }
        elif self.mode == 'valid':
            file_idx = idx // (self.batchsize // self.windowlength)
            if offset is None:
                offset = (idx % (self.batchsize // self.windowlength)) * self.windowlength
            return {
                'file_path': self.files_path[file_idx],
                'file_idx': file_idx,
                'sample_idx': idx,
                'frame_offset': offset,
                'mode': self.mode
            }
        else:
            # For other modes or unknown cases
            return {
                'file_path': 'unknown',
                'file_idx': -1,
                'sample_idx': idx,
                'frame_offset': 0,
                'mode': self.mode or 'unknown'
            }

# --- code that is specific for a dataset ---
class HSEDataset(BaseHSEDataset):
    def __init__(self, dataset_dir, rellist_path, config, mode=None):
        super().__init__(dataset_dir, rellist_path, config, mode=mode)
        if mode not in ['train', 'valid']:
            raise ValueError("Mode must be 'train' or 'valid'")

    def __len__(self):
        if self.mode == 'train':
            return self.file_num
        elif self.mode == 'valid':
            return self.file_num * (self.batchsize // self.windowlength)

    def __getitem__(self, idx):
        if self.mode == 'train':
            offset = random.randint(0, self.batchsize - self.windowlength - 1)
            ind_frm = idx * self.batchsize + offset
            file_info = self._get_file_info(idx, offset)
            return self.x[:, ind_frm:ind_frm+self.windowlength, :], self.y[0, ind_frm:ind_frm+self.windowlength, :].unsqueeze(0), file_info
        elif self.mode == 'valid':
            file_idx = idx // (self.batchsize // self.windowlength)
            offset = (idx % (self.batchsize // self.windowlength)) * self.windowlength
            ind_frm = file_idx * self.batchsize + offset
            file_info = self._get_file_info(idx, offset)
            return self.x[:, ind_frm:ind_frm+self.windowlength, :], self.y[0, ind_frm:ind_frm+self.windowlength, :].unsqueeze(0), file_info

class HSEfe4_dataset(BaseHSEDataset):
    def __init__(self, dataset_dir, rellist_path, config, mode=None):
        # HSEfe4 uses windowlength = 2*contextview+1
        dataset_config = config.get('dataset', {})
        contextview = dataset_config.get('contextview', 3)
        modified_config = config.copy()
        modified_config['dataset'] = dataset_config.copy()
        modified_config['dataset']['windowlength'] = 2 * contextview + 1
        super().__init__(dataset_dir, rellist_path, modified_config, mode=mode)

    def __len__(self):
        return self.y.size(1)

    def __getitem__(self, idx):
        idx = idx % (self.file_num * self.batchsize)
        if idx > self.file_num * self.batchsize - self.contextview - 1:
            idx = self.file_num * self.batchsize - self.contextview - 1
        if idx < self.contextview:
            idx = self.contextview
        
        # Calculate file info for HSEfe4 dataset
        file_idx = idx // self.batchsize
        frame_offset = idx % self.batchsize
        file_info = {
            'file_path': self.files_path[file_idx] if file_idx < len(self.files_path) else 'unknown',
            'file_idx': file_idx,
            'sample_idx': idx,
            'frame_offset': frame_offset,
            'mode': self.mode or 'unknown'
        }
        
        return self.x[:, idx-self.contextview:idx+self.contextview+1, :], self.y[0, idx-self.contextview:idx+self.contextview+1, :].unsqueeze(0), file_info

class HSELPSDataset(BaseHSEDataset):
    def __init__(self, dataset_dir, rellist_path, config, mode=None):
        # HSELPS uses windowlength = 400 and use_lambda = True by default
        dataset_config = config.get('dataset', {})
        modified_config = config.copy()
        modified_config['dataset'] = dataset_config.copy()
        modified_config['dataset']['windowlength'] = dataset_config.get('windowlength', 400)
        modified_config['dataset']['use_lambda'] = dataset_config.get('use_lambda', True)
        super().__init__(dataset_dir, rellist_path, modified_config, mode=mode)

    def __len__(self):
        return self.y.size(1)

    def __getitem__(self, idx):
        idx = idx % (self.file_num * self.batchsize)
        fileidx = idx // self.batchsize
        offset = idx % self.batchsize
        if offset > self.batchsize - self.windowlength - 1:
            offset = self.batchsize - self.windowlength - 1
        dataidx = fileidx * self.batchsize + offset
        
        # File info for HSELPS dataset
        file_info = {
            'file_path': self.files_path[fileidx] if fileidx < len(self.files_path) else 'unknown',
            'file_idx': fileidx,
            'sample_idx': idx,
            'frame_offset': offset,
            'mode': self.mode or 'unknown'
        }
        
        return self.x[:, dataidx:dataidx+self.windowlength, :], self.y[0, dataidx:dataidx+self.windowlength, :].unsqueeze(0), self.truelambda[fileidx], file_info
