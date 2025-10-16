import torch
from scipy.signal import get_window
from torch import istft
from rfc_utils import check_cuda_for_bss_eval, hse_eval


class EvaluationMetrics:
    def __init__(self, file_num, eval_extended=False, eval_params=None):
        self.eval_extended = eval_extended
        self.SNR = torch.zeros(file_num, 1)
        self.SNR_mix = torch.zeros(file_num, 1)
        
        if eval_extended and eval_params:
            self.nfft = eval_params.get('nfft', 512)
            self.hop_length = eval_params.get('hop_length', 128)
            self.window_name = eval_params.get('window_name', 'hamming')
            self.chan_ref = eval_params.get('chan_ref', 0)
            self.fs = eval_params.get('fs', 16000)
            
            self.sdr_bss = torch.zeros(file_num, 2)
            self.sir_bss = torch.zeros(file_num, 2)
            self.sar_bss = torch.zeros(file_num, 2)
            self.stoi = torch.zeros(file_num, 2)
            self.sdr_td = torch.zeros(file_num, 2)
            self.window = None  # Will be set when device is known
    
    def setup_device(self, device):
        if self.eval_extended:
            self.window = torch.from_numpy(get_window(self.window_name, self.nfft)).to(device)
            # Ensure CUDA is available for evaluation
            device = check_cuda_for_bss_eval()
        self.device = device  # Store device for later use
        return device
    
    def compute_basic_metrics(self, lpF, s_out, v_out, s, v):
        self.SNR[lpF, 0] = 10 * torch.log10(torch.mean(torch.abs(s_out) ** 2, dim=(1, 3)) / torch.mean(torch.abs(v_out) ** 2, dim=(1, 3)))
        chan_ref = getattr(self, 'chan_ref', 0)
        self.SNR_mix[lpF, 0] = 10 * torch.log10(torch.mean(torch.abs(s[:, :, chan_ref:chan_ref+1, :]) ** 2, dim=(1, 3)) / torch.mean(torch.abs(v[:, :, chan_ref:chan_ref+1, :]) ** 2, dim=(1, 3)))
    
    def compute_extended_metrics(self, lpF, x, s, v, s_out, w_out):
        if not self.eval_extended:
            return
            
        x = x.permute(0, 3, 1, 2)
        s_hat = torch.matmul(w_out.conj().transpose(2, 3), x)
        
        # Convert to time domain
        s_hat_td = istft(s_hat.squeeze(2), n_fft=self.nfft, hop_length=self.hop_length, win_length=self.nfft, window=self.window)
        s_true_td = istft(s[:, :, self.chan_ref, :].squeeze(2), n_fft=self.nfft, hop_length=self.hop_length, win_length=self.nfft, window=self.window)
        v_true_td = istft(v[:, :, self.chan_ref, :].squeeze(2), n_fft=self.nfft, hop_length=self.hop_length, win_length=self.nfft, window=self.window)
        x_true_td = istft(x[:, :, self.chan_ref, :].squeeze(2), n_fft=self.nfft, hop_length=self.hop_length, win_length=self.nfft, window=self.window)
        s_original_td = istft(s_out.squeeze(2), n_fft=self.nfft, hop_length=self.hop_length, win_length=self.nfft, window=self.window)
        
        # Calculate metrics
        self.sdr_bss[lpF, 0], self.sir_bss[lpF, 0], self.sar_bss[lpF, 0], self.stoi[lpF, 0], self.sdr_td[lpF, 0] = hse_eval(x_true_td, torch.cat([s_true_td, v_true_td], dim=0), s_true_td, fs=self.fs)
        self.sdr_bss[lpF, 1], self.sir_bss[lpF, 1], self.sar_bss[lpF, 1], self.stoi[lpF, 1], self.sdr_td[lpF, 1] = hse_eval(s_hat_td, torch.cat([s_true_td, v_true_td], dim=0), s_original_td, fs=self.fs)
    
    def get_results_dict(self):
        results = {
            'SNR': self.SNR.cpu().numpy(),
            'SNR_mix': self.SNR_mix.cpu().numpy()
        }
        
        if self.eval_extended:
            results.update({
                'SDR_bss': self.sdr_bss.cpu().numpy(),
                'SIR_bss': self.sir_bss.cpu().numpy(),
                'SAR_bss': self.sar_bss.cpu().numpy(),
                'STOI': self.stoi.cpu().numpy(),
                'SDR_td': self.sdr_td.cpu().numpy()
            })
        
        return results
