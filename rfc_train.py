import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch
import gc
import os
import shutil
import argparse

from rfc_utils import load_json_config, CustomLRLogger

# Import all possible dataset and model classes
from rfc_datasets import HSEDataset, HSEfe4_dataset
from rfc_nn import  HSE4_joint2,  HSEfe4_nad3 


def cleanup_training_session():
    """Clean up memory and resources after training session"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --- Helper to select dataset/model by config or argument ---
def get_dataset_and_model(config):
    mode = config.get('mode', 'HSE4')
    if mode == 'HSEfe4_nad3':
        dataset_cls = HSEfe4_dataset
        model_cls = HSEfe4_nad3
    elif mode == 'HSE4_joint2':
        dataset_cls = HSEDataset
        model_cls = HSE4_joint2
    else:
        raise ValueError(f"Unknown mode: {mode}.")
    return dataset_cls, model_cls

# --- Main training logic ---
def train_model(config_path):
    config = load_json_config(config_path)

    seed = config.get('seed', 42)
    
    # Set PyTorch seeds first for deterministic behavior
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Then set seeds for all other libraries (numpy, random, etc.)
    pl.seed_everything(seed, workers=True)
    
    print(f"Random seed set to: {seed}")

    dataset_train_path = config['paths'].get('train_path', None)
    dataset_train_rellist = config['paths'].get('train_rellist', None)
    dataset_valid_path = config['paths'].get('valid_path', None)
    dataset_valid_rellist = config['paths'].get('valid_rellist', None)
    fe_path = config['paths'].get('fe_path', None)
    hse_path = config['paths'].get('hse_path', None)
    vername = config.get('version', None)
    odir_logs = config['paths'].get('odir_logs', None)
    odir_chcks = os.path.join(odir_logs, 'lightning_logs', vername, "checkpoints")

    epochs = config['train'].get('epochs', None)
    batch_train = config['train'].get('batch_train', None)
    batch_valid = config['train'].get('batch_valid', None)
    stop_delta = config['train'].get('stopping_delta', None)
    stop_pati = config['train'].get('stopping_patience', None)
    val_check_int = config['train'].get('val_check_interval', None)
    grad_clip = config['train'].get('grad_clip', None)
    log_every_n_steps = config['train'].get('log_every_n_steps', None)
    
    
    dataset_cls, model_cls = get_dataset_and_model(config)

    HSEdatasettrain = dataset_cls(dataset_dir=dataset_train_path, rellist_path=dataset_train_rellist, config=config, mode='train')
    HSEdatasetval = dataset_cls(dataset_dir=dataset_valid_path, rellist_path=dataset_valid_rellist, config=config, mode='valid')

    train_dataloader = DataLoader(HSEdatasettrain, batch_size=batch_train, shuffle=True)
    val_dataloader = DataLoader(HSEdatasetval, batch_size=batch_valid, shuffle=False)

    tb_logger = TensorBoardLogger(save_dir=odir_logs, name="lightning_logs", version=vername)

    os.makedirs(odir_chcks, exist_ok=True)
    shutil.copy(config_path, odir_chcks)

    if fe_path is None: # HSEfe4
        mynet = model_cls(config=config)
    elif hse_path != '': # HSE4
        mynet = model_cls.load_from_checkpoint(hse_path, config=config)
    else: # HSE4
        mynet = model_cls.load_from_checkpoint(fe_path, config=config)

    checkpoint_callback = ModelCheckpoint(dirpath=odir_chcks, save_top_k=1, monitor="valid_loss", mode="min")
    lr_monitor = CustomLRLogger() # HSE4
    # lr_monitor = pl.callbacks.LearningRateMonitor() # HSEfe4
    stopping_callback = EarlyStopping(monitor="valid_loss", min_delta=stop_delta, patience=stop_pati, verbose=True, mode="min")

    trainer = pl.Trainer(logger=tb_logger, max_epochs=epochs, accelerator="gpu",
                        devices=1, log_every_n_steps=log_every_n_steps, val_check_interval=val_check_int,
                        gradient_clip_val=grad_clip,
                        callbacks=[checkpoint_callback, lr_monitor, stopping_callback])
    trainer.fit(mynet, train_dataloader, val_dataloader)
    
    print("Cleaning up training session...")
    
    # Close logger properly
    if hasattr(trainer.logger, 'finalize'):
        trainer.logger.finalize("success")
    if hasattr(trainer.logger, 'experiment') and trainer.logger.experiment:
        trainer.logger.experiment.close()
    
    del mynet
    del trainer
    del train_dataloader
    del val_dataloader
    del HSEdatasettrain
    del HSEdatasetval
    del tb_logger
    del checkpoint_callback
    del lr_monitor
    del stopping_callback
    
    # Force cleanup
    cleanup_training_session()
    
    print('Done!!')

def main():    
    parser = argparse.ArgumentParser(description='Train HSE model with specified config(s)')
    parser.add_argument('--config', type=str, nargs='+', default=['HSE4_train.json'], 
                       help='Path(s) to config JSON file(s). Can specify multiple files. (default: HSE4_train.json)')
    parser.add_argument('--folder', type=str, 
                       help='Path to folder containing config files. Will train with all .json files in the folder.')
    
    args = parser.parse_args()
    
    if args.folder:
        folder_path = args.folder
        if not os.path.isabs(folder_path):
            # If relative path, make it relative to the script directory
            folder_path = os.path.join(os.path.dirname(__file__), folder_path)
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder not found: {folder_path}")
            return
        
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
        config_files = args.config if isinstance(args.config, list) else [args.config]
    
    print(f"Found {len(config_files)} config file(s) to process")
    
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
        
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            print("Skipping this config and continuing with the next one...")
            missing_configs.append(os.path.basename(config_file) if args.folder else config_file)
            continue
        
        print(f"Using config: {config_path}")
        
        try:
            train_model(config_path=config_path)
            print(f"Successfully completed training with {os.path.basename(config_file) if args.folder else config_file}")
            successful_configs.append(os.path.basename(config_file) if args.folder else config_file)

            cleanup_training_session()
        except Exception as e:
            print(f"Error during training with {os.path.basename(config_file) if args.folder else config_file}: {e}")
            print("Continuing with next config...")
            failed_configs.append((os.path.basename(config_file) if args.folder else config_file, str(e)))

            cleanup_training_session()
            continue
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
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
        print("All training sessions completed successfully!")
    else:
        print("Some training sessions encountered issues. Check the details above.")
    print(f"{'='*60}")

if __name__ == "__main__":
    # # Use relative paths from the current file's directory
    # config_path_hsefe4 = os.path.join(os.path.dirname(__file__), 'HSEfe4_train.json')
    # config_path_hse4 = os.path.join(os.path.dirname(__file__), 'HSE4_train.json')

    # main(config_path=config_path_hse4)
    main()