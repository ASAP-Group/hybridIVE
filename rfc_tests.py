import os
import sys
import shutil
import numpy as np
import torch
from torch import istft
from scipy.io import savemat
import argparse

from rfc_utils import load_json_config, load_mat_file, load_file_list, hse_eval, check_cuda_for_bss_eval
from rfc_EvalMetrics import EvaluationMetrics
from rfc_blocks import mcwf_filter
from rfc_showres import show_single_algorithm

from rfc_nn import HSE4_joint2, HSEfe4_nad3

def get_test_function(mode):
    """Get the appropriate test function based on mode"""
    if mode == 'HSE4_joint2':
        return run_hse4_joint2_test

    # All HSEfe4_nad3 variants go through one unified runner
    elif mode in ('HSEfe4_nad3', 'HSEfe4_nad3_oracle', 'HSEfe4_nad3_blind'):
        return (lambda config_path, m=mode: run_hsefe4_nad3_test_unified(config_path, m))

    elif mode == 'MCWF':
        return run_mcwf_test
    else:
        raise ValueError(f"Unknown mode: {mode}.")

def run_test_generic(config_test_path, model_loader, forward_fn, set_numit_fn=None):
    config_test = load_json_config(config_test_path)
    
    eval_extended = config_test.get('eval_extended', False)
    eval_params = config_test.get('eval_extended_params') if eval_extended else None
    
    # Load configuration and model
    dataset_dir = config_test.get('paths', {}).get('test_path', None)
    dataset_test_rellist = config_test.get('paths', {}).get('test_rellist', None)
    odir_path = config_test.get('paths', {}).get('odir_path', None)
    odir_name = config_test.get('paths', {}).get('odir_name', None)
    model_path = config_test.get('checkpoint', {}).get('chck_path', None)
    config_train_path = config_test.get('checkpoint', {}).get('config_train_path', None)
    config_train = load_json_config(config_train_path)
    
    try:
        model = model_loader(model_path, config_train)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print(f"Config train keys: {list(config_train.keys())}")
        raise e
    model.eval()
    if set_numit_fn is not None:
        set_numit_fn(model, config_test)
    
    files_path = load_file_list(dataset_test_rellist)
    file_num = len(files_path)
    
    # Initialize metrics handler
    metrics = EvaluationMetrics(file_num, eval_extended, eval_params)
    device = metrics.setup_device(model.device) 
    model = model.to(device)

    # Main processing loop
    for lpF in range(file_num):
        if np.remainder(lpF, 10) == 0:
            print(f'Loading file {lpF} / {file_num}')
        file_path = os.path.join(dataset_dir, files_path[lpF])
        mat = load_mat_file(file_path)
        x = torch.from_numpy(mat['x']).detach().to(dtype=torch.complex64).to(device=device)
        s = torch.from_numpy(mat['y']).detach().to(dtype=torch.complex64).to(device=device)
        x = x[None, :, :, :]
        s = s[None, :, :, :]
        v = x - s
        s = s.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            s_out, v_out, w_out = forward_fn(model, x, s, v)
        
        metrics.compute_basic_metrics(lpF, s_out, v_out, s, v)
        
        metrics.compute_extended_metrics(lpF, x, s, v, s_out, w_out)

    # Save results
    odir_path_curr = os.path.join(odir_path, odir_name)
    os.makedirs(odir_path_curr, exist_ok=True)
    shutil.copy(config_test_path, odir_path_curr)
    shutil.copy(config_train_path, odir_path_curr)
    
    results = metrics.get_results_dict()
    results_mat_path = os.path.join(odir_path_curr, f'results_{odir_name}.mat')
    savemat(results_mat_path, results)
    
    # Generate and save summary of computed metrics
    summary_txt_path = os.path.join(odir_path_curr, f'summary_{odir_name}.txt')
    show_single_algorithm(results_mat_path, odir_name, 
                         eval_extended=eval_extended, 
                         show_improvements=False,
                         output_file=summary_txt_path)
    
    print('Done!!')

# --- Test functions for different models ---
def run_hse4_joint2_test(config_test_path):
    def model_loader(model_path, config_train):
        return HSE4_joint2.load_from_checkpoint(model_path, config=config_train)
    def forward_fn(model, x, s, v):
        _, w_finetuned = model.forward(x)
        s_finetuned = torch.matmul(w_finetuned.conj().transpose(2, 3), s)
        v_finetuned = torch.matmul(w_finetuned.conj().transpose(2, 3), v)
        return s_finetuned, v_finetuned, w_finetuned
    def set_numit_fn(model, config_test):
        test_numit = config_test.get('test', {}).get('test_numit', None)
        if test_numit is not None:
            model.numit = test_numit
    run_test_generic(config_test_path, model_loader, forward_fn, set_numit_fn=set_numit_fn)

def run_hsefe4_nad3_test_unified(config_test_path, mode):
    """
    mode ∈ {'HSEfe4_nad3', 'HSEfe4_nad3_oracle', 'HSEfe4_nad3_blind'}
    maps to test_forward(..., mode={'learned'|'oracle'|'blind'})
    """
    def model_loader(model_path, config_train):
        return HSEfe4_nad3.load_from_checkpoint(model_path, config=config_train)

    def forward_fn(model, x, s, v):
        # map CLI mode to model.new_test_forward mode
        if mode == 'HSEfe4_nad3':
            nf_mode = 'learned'
            y = None
        elif mode == 'HSEfe4_nad3_oracle':
            nf_mode = 'oracle'
            y = s                       # oracle uses ground-truth s as 'y'
        elif mode == 'HSEfe4_nad3_blind':
            nf_mode = 'blind'
            y = s                       # blind uses ground-truth s as 'y' to know the dimension of ones_like tensor
        else:
            raise ValueError(f"Unexpected mode: {mode}")

        w = model.test_forward(x, y=y, mode=nf_mode)

        # project sources/noise the same way as before
        s_out = torch.matmul(w.conj().transpose(2, 3), s)
        v_out = torch.matmul(w.conj().transpose(2, 3), v)
        return s_out, v_out, w

    def set_numit_fn(model, config_test):
        test_numit = config_test.get('test', {}).get('test_numit', None)
        if test_numit is not None:
            model.numit = test_numit

    run_test_generic(config_test_path, model_loader, forward_fn, set_numit_fn=set_numit_fn)

def run_mcwf_test(config_test_path):
    """
    MCWF (Multi-channel Wiener Filter) test - No neural network model needed.
    Uses classic signal processing algorithm from rfc_blocks.mcwf_filter.
    Leverages existing utilities from rfc_utils and new MCWF-specific functions in rfc_evalMetrics.
    
    Config parameters:
    - bss_eval (bool, default=False): Enable BSS evaluation metrics (SDR_bss, SIR_bss, SAR_bss, STOI, SDR_td).
      Requires CUDA-enabled GPU for fast_bss_eval library.
    """
    config_test = load_json_config(config_test_path)
    
    eval_extended = config_test.get('eval_extended', False)
    eval_params = config_test.get('eval_extended_params') if eval_extended else None
    show_improvements = config_test.get('show_improvements', False)
    bss_eval = config_test.get('bss_eval', False)
    
    # Load configuration
    dataset_dir = config_test['paths']['test_path']
    dataset_test_rellist = config_test['paths']['test_rellist']
    odir_path = config_test['paths']['odir_path']
    odir_name = config_test['paths']['odir_name']
    
    # MCWF specific parameters
    nfft = config_test['test']['nfft']
    ffthop = config_test['test']['ffthop']
    fs = config_test['test']['fs']
    
    # Update eval_params with MCWF parameters if needed
    if eval_extended and eval_params:
        eval_params.update({
            'nfft': nfft,
            'hop_length': ffthop,
            'fs': fs
        })
    
    files_path = load_file_list(dataset_test_rellist)
    file_num = len(files_path)
    
    # Initialize metrics handler - reuse existing utility
    metrics = EvaluationMetrics(file_num, eval_extended, eval_params)
    device = metrics.setup_device(torch.device('cpu'))  # MCWF uses CPU by default
    
    # Main processing loop
    for lpF in range(file_num):
        if np.remainder(lpF, 10) == 0:
            print(f'Loading file {lpF} / {file_num}')
        file_path = os.path.join(dataset_dir, files_path[lpF])
        
        mat = load_mat_file(file_path)
        
        # Load data in original format [M,L,K]
        x_orig = torch.from_numpy(mat['x']).detach().to(dtype=torch.complex64)
        s_orig = torch.from_numpy(mat['y']).detach().to(dtype=torch.complex64)
        v_orig = x_orig - s_orig
        
        # Apply MCWF filter (operates on original tensor format)
        _, w = mcwf_filter(x_orig, s_orig, nfft, ffthop, fs)
        
        # Convert to tensor and adapt to expected format for MCWF metrics
        w_tensor = torch.from_numpy(w).detach().to(dtype=torch.complex64)  # [M, K]
        w_tensor = w_tensor.permute(1, 0).unsqueeze(1)  # [K, 1, M] 
        
        # Convert data to expected format for MCWF metrics [K, M, L] 
        x = x_orig.permute(2, 0, 1)  # [K, M, L]
        s = s_orig.permute(2, 0, 1)  # [K, M, L]
        v = v_orig.permute(2, 0, 1)  # [K, M, L]
        
        # Apply filter to get outputs
        s_original = torch.matmul(w_tensor.conj(), s)  # [K, 1, L]
        v_original = torch.matmul(w_tensor.conj(), v)  # [K, 1, L]
        
        # Transform dimensions to match regular metrics interface: [K, M, L] → [1, K, M, L]
        s_out_transformed = s_original.unsqueeze(0)  # [K, 1, L] → [1, K, 1, L]
        v_out_transformed = v_original.unsqueeze(0)  # [K, 1, L] → [1, K, 1, L]
        s_transformed = s.unsqueeze(0)               # [K, M, L] → [1, K, M, L]
        v_transformed = v.unsqueeze(0)               # [K, M, L] → [1, K, M, L]
        
        # Compute metrics using unified functions
        metrics.compute_basic_metrics(lpF, s_out_transformed, v_out_transformed, s_transformed, v_transformed)
        
        # BSS evaluation (controlled by bss_eval flag)
        if bss_eval:
            # Check CUDA availability for BSS evaluation
            try:
                bss_device = check_cuda_for_bss_eval()
            except RuntimeError as e:
                print(f"Warning: BSS evaluation requires CUDA. Skipping BSS metrics. Error: {e}")
                bss_eval = False  # Disable for remaining files
            else:
                # Set up window for ISTFT
                window = torch.hann_window(nfft).to(bss_device)
                chan_ref = 0  # Reference channel
                
                # Move tensors to BSS device before processing
                x_bss = x.to(bss_device)
                s_bss = s.to(bss_device)
                v_bss = v.to(bss_device)
                s_original_bss = s_original.to(bss_device)
                w_tensor_bss = w_tensor.to(bss_device)
                
                # Apply filter to get enhanced signal
                s_hat = torch.matmul(w_tensor_bss.conj(), x_bss)  # [K, 1, L]
                
                # Convert to time domain for BSS evaluation
                s_hat_td = istft(s_hat.squeeze(1), n_fft=nfft, hop_length=ffthop, 
                                win_length=nfft, window=window).unsqueeze(0)
                s_true_td = istft(s_bss[:, chan_ref, :], n_fft=nfft, hop_length=ffthop, 
                                 win_length=nfft, window=window).unsqueeze(0)
                v_true_td = istft(v_bss[:, chan_ref, :], n_fft=nfft, hop_length=ffthop, 
                                 win_length=nfft, window=window).unsqueeze(0)
                x_true_td = istft(x_bss[:, chan_ref, :], n_fft=nfft, hop_length=ffthop, 
                                 win_length=nfft, window=window).unsqueeze(0)
                s_original_td = istft(s_original_bss.squeeze(1), n_fft=nfft, hop_length=ffthop, 
                                     win_length=nfft, window=window).unsqueeze(0)
                
                # Initialize BSS metrics storage if not already done
                if not hasattr(metrics, 'sdr_bss'):
                    metrics.sdr_bss = torch.zeros((file_num, 2))
                    metrics.sir_bss = torch.zeros((file_num, 2))
                    metrics.sar_bss = torch.zeros((file_num, 2))
                    metrics.stoi = torch.zeros((file_num, 2))
                    metrics.sdr_td = torch.zeros((file_num, 2))
                
                # Calculate BSS metrics
                metrics.sdr_bss[lpF, 0], metrics.sir_bss[lpF, 0], metrics.sar_bss[lpF, 0], metrics.stoi[lpF, 0], metrics.sdr_td[lpF, 0] = hse_eval(
                    x_true_td, torch.cat([s_true_td, v_true_td], dim=0), s_true_td, fs=fs)
                metrics.sdr_bss[lpF, 1], metrics.sir_bss[lpF, 1], metrics.sar_bss[lpF, 1], metrics.stoi[lpF, 1], metrics.sdr_td[lpF, 1] = hse_eval(
                    s_hat_td, torch.cat([s_true_td, v_true_td], dim=0), s_original_td, fs=fs)
        
        # Transform dimensions for extended metrics: [K, M, L] → [1, K, M, L]
        x_transformed = x.unsqueeze(0)                # [K, M, L] → [1, K, M, L]
        # s_transformed = s.unsqueeze(0)                # [K, M, L] → [1, K, M, L] 
        # v_transformed = v.unsqueeze(0)                # [K, M, L] → [1, K, M, L]
        # s_out_transformed = s_original.unsqueeze(0)   # [K, 1, L] → [1, K, 1, L]
        w_out_transformed = w_tensor.unsqueeze(0)     # [K, 1, M] → [1, K, 1, M]
        
        x_transformed_permuted = x_transformed.permute(0,2,3,1)
        w_out_transformed_permuted = w_out_transformed.permute(0,1,3,2)
        # Compute extended metrics using unified function
        metrics.compute_extended_metrics(lpF, x_transformed_permuted, s_transformed, v_transformed, s_out_transformed, w_out_transformed_permuted)
    
    # Save results using existing utility
    odir_path_curr = os.path.join(odir_path, odir_name)
    os.makedirs(odir_path_curr, exist_ok=True)
    shutil.copy(config_test_path, odir_path_curr)
    
    results = metrics.get_results_dict()
    
    # Add BSS evaluation results if enabled
    if bss_eval and hasattr(metrics, 'sdr_bss'):
        results.update({
            'SDR_bss': metrics.sdr_bss.cpu().numpy(),
            'SIR_bss': metrics.sir_bss.cpu().numpy(),
            'SAR_bss': metrics.sar_bss.cpu().numpy(),
            'STOI': metrics.stoi.cpu().numpy(),
            'SDR_td': metrics.sdr_td.cpu().numpy()
        })
    
    results_mat_path = os.path.join(odir_path_curr, f'results_{odir_name}.mat')
    savemat(results_mat_path, results)
    
    # Generate and save summary of computed metrics
    summary_txt_path = os.path.join(odir_path_curr, f'summary_{odir_name}.txt')
    show_single_algorithm(results_mat_path, odir_name, 
                         eval_extended=eval_extended, 
                         show_improvements=show_improvements,
                         output_file=summary_txt_path)
    
    print('Done!!')

def main():
    parser = argparse.ArgumentParser(description='Run RFC tests for HSE models with specified config(s)')
    parser.add_argument('--mode', type=str, required=True, choices=['HSE4_joint2', 'HSEfe4_nad3', 'HSEfe4_nad3_oracle', 'HSEfe4_nad3_blind', 'MCWF'], 
                       help='Test mode to run')
    parser.add_argument('--config', type=str, nargs='+',
                       help='Path(s) to test config JSON file(s). Can specify multiple files.')
    parser.add_argument('--folder', type=str, 
                       help='Path to folder containing config files. Will test with all .json files in the folder.')
    
    args = parser.parse_args()
    
    # Handle folder argument
    if args.folder:
        folder_path = args.folder
        if not os.path.isabs(folder_path):
            # If relative path, make it relative to the script directory
            folder_path = os.path.join(os.path.dirname(__file__), folder_path)
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder not found: {folder_path}")
            return
        
        # Find all .json files in the folder
        config_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                config_files.append(os.path.join(folder_path, file))
        
        if not config_files:
            print(f"No .json config files found in folder: {folder_path}")
            return
        
        # Sort for consistent ordering
        config_files.sort()
        print(f"Found {len(config_files)} config files in folder: {folder_path}")
        for config_file in config_files:
            print(f"  - {os.path.basename(config_file)}")
    else:
        if not args.config:
            print("Error: Either --config or --folder must be specified")
            return
        # Convert single config to list for uniform processing
        config_files = args.config if isinstance(args.config, list) else [args.config]
    
    print(f"Found {len(config_files)} config file(s) to process")
    print(f"Test mode: {args.mode}")
    
    # Get the appropriate test function
    test_fn = get_test_function(args.mode)
    
    # Track results for final summary
    successful_configs = []
    failed_configs = []
    missing_configs = []
    
    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing config {i}/{len(config_files)}: {os.path.basename(config_file) if args.folder else config_file}")
        print(f"{'='*60}")
        
        # Handle both absolute and relative paths
        if os.path.isabs(config_file) or args.folder:
            config_path = config_file
        else:
            # First try in configs subdirectory
            config_path = os.path.join(os.path.dirname(__file__), 'configs', config_file)
            # If not found, try in the same directory as the script
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(__file__), config_file)
        
        # Check if config file exists
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            print("Skipping this config and continuing with the next one...")
            missing_configs.append(os.path.basename(config_file) if args.folder else config_file)
            continue
        
        print(f"Using config: {config_path}")
        
        try:
            # Use the test function directly
            test_fn(config_path)
            print(f"Successfully completed testing with {os.path.basename(config_file) if args.folder else config_file}")
            successful_configs.append(os.path.basename(config_file) if args.folder else config_file)
        except Exception as e:
            print(f"Error during testing with {os.path.basename(config_file) if args.folder else config_file}: {e}")
            print("Continuing with next config...")
            failed_configs.append((os.path.basename(config_file) if args.folder else config_file, str(e)))
            continue
    
    # Print final summary
    print(f"\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    
    print(f"Total configs processed: {len(config_files)}")
    print(f"Successful: {len(successful_configs)}")
    print(f"Failed: {len(failed_configs)}")
    print(f"Missing: {len(missing_configs)}")
    
    if successful_configs:
        print(f"\n SUCCESSFUL CONFIGS ({len(successful_configs)}):")
        for config in successful_configs:
            print(f"  - {config}")
    
    if missing_configs:
        print(f"\n MISSING CONFIGS ({len(missing_configs)}):")
        for config in missing_configs:
            print(f"  - {config}")
    
    if failed_configs:
        print(f"\n FAILED CONFIGS ({len(failed_configs)}):")
        for config, error in failed_configs:
            print(f"  - {config}: {error}")
    
    print(f"\n{'='*60}")
    if len(failed_configs) == 0 and len(missing_configs) == 0:
        print("All testing sessions completed successfully!")
    else:
        print("Some testing sessions encountered issues. Check the details above.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
