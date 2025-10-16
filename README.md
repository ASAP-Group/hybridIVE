# From Informed Independent Vector Extraction to Hybrid Architectures for Target Source Extraction

This repository corresponds to a publication named "From Informed Independent Vector Extraction to
Hybrid Architectures for Target Source Extraction".


## Project Structure

```
hybridIVE/
├── .vscode
│   └── launch.json                                             # VScode examples of usage
├── IFIVAModule.py                                              # Standalone iFIVA implementation module
├── rfc_nn.py                                                   # Neural network models (HSE4, HSEfe4, etc.)
├── rfc_train.py                                                # Training pipeline with PyTorch Lightning
├── rfc_datasets.py                                             # Dataset loaders for multichannel audio data
├── rfc_utils.py                                                # Utility functions and robust operations
├── rfc_blocks.py                                               # Neural network building blocks
├── rfc_EvalMetrics.py                                          # Evaluation metrics and scoring functions
├── rfc_tests.py                                                # Test suite and validation scripts
├── rfc_showres.py                                              # Results visualization and analysis
├── configs/                                                    # Configuration files for training/testing
│   ├── train/                                                  # Training configurations
│   └── test/                                                   # Testing configurations
├── dataset_matlab/                                             # MATLAB scripts for dataset generation
│   ├── wsj0_files_to_used_dataset.m                            # Helper function for dataset generation
│   ├── generate_dataset.m                                      # Main dataset generation script
│   └── make_dataset.m                                          # Dataset creation wrapper
├── simulations/                                                # Matlab files for simulations from Section V and the experiment from Section VI.E
│   └── *.m
└── lightning_logs/                                             # Training logs and checkpoints
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Lightning
- TensorBoard
- NumPy
- SciPy
- MATLAB (for dataset generation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ASAP-Group/HSE_ainfo.git
cd HSE_ainfo
```

2. Install Python dependencies:
```bash
pip install torch pytorch-lightning tensorboard numpy scipy
pip install fast-bss-eval torchmetrics
pip install mat73  # For loading MATLAB v7.3 files
```

3. For dataset generation, ensure MATLAB is installed with the RIR Generator:
   - Download: https://github.com/ehabets/RIRu-Generator
   - Install: https://www.audiolabs-erlangen.de/fau/professor/habets/software/rir-generator

## Usage

### Dataset Generation
Before you generate the data for training you will need to prepare the wavs.
We selected 3557 recordings at random. These files were used as described below ("path/to/wsj").

For example of file handling from original wsj0 dataset please check [`dataset_matlab/wsj0_files_to_used_dataset.m`](dataset_matlab/wsj0_files_to_used_dataset.m)

There is a need to convert the original licensed wsj0 dataset from `.wv1` into `.wav` files.
You may do so by using the matlab toolbox http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html more specifically the `v_readsph.m` function.

The training/test datasets are generated using dry files from the licensed WSJ0 dataset. We selected randomly 3557 file, whose paths are stored in the file [`dataset_matlab/used_wavs.mat`](dataset_matlab/used_wavs.mat). Use the script [`dataset_matlab/wsj0_files_to_used_dataset.m`](dataset_matlab/wsj0_files_to_used_dataset.m) to copy the selected files from the original WSJ0 structure to a single directory.

The original WSJ0 files need to be converted from `.wv1` format into `.wav` files. You may do so by using the matlab toolbox http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html more specifically the `v_readsph.m` function.

Subsequently, use the dry `.wav` files to generate the train/test data through the MATLAB scripts:


```matlab
% Basic dataset generation
generate_dataset(1500, "path/to/output/", "path/to/wsj/", 42);

% Advanced generation with custom parameters
generate_dataset(1500, "path/to/output/", "path/to/wsj/", 42, [], 1000, 200, 300);
```

Specific examples are given in [`dataset_matlab/make_dataset.m`](dataset_matlab/make_dataset.m)

### Training

Configure and train models using the provided configuration files:

```bash
# Train NAD model
python rfc_train.py --config configs/train/NAD_train_baseline_scenario.json

# Train HSE4 model
python rfc_train.py --config configs/train/Fine_tuned_NAD_train_baseline_scenario.json
```

### Testing and Evaluation

Evaluate trained models on test datasets:

```bash
# Test model performance
python rfc_tests.py --mode "HSE4_joint2" --config configs/test/Fine_tuned_NAD_test_baseline_scenario.json

# Show detailed results separately
python rfc_showres.py --results_dir lightning_logs/experiment_name/
```

## Evaluation Metrics

The system reports multiple standard metrics:

- **SDR** (Signal-to-Distortion Ratio): Overall separation quality
- **SIR** (Signal-to-Interference Ratio): Interference suppression
- **SAR** (Signal-to-Artifacts Ratio): Processing artifacts
- **eSTOI** (Short-Time Objective Intelligibility): Speech intelligibility - 

## License

TBD

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hybridive,
  title={From Informed Independent Vector Extraction to Hybrid Architectures for Target Source Extraction},
  author={Zbynek Koldovsky, Jirı Male∗, Martin Vratny, Tereza Vrbova, Jaroslav Cmejla, and Stephen O’Regan},
  year={2025},
  url={https://github.com/ASAP-Group/hybridIVE}
}
```

## References

- [eSTOI](https://ieeexplore.ieee.org/abstract/document/7539284) - extended short-time objective intelligibility.
- [fast_bss_eval](https://github.com/fakufaku/fast_bss_eval) - fast implementation of the bss_eval metrics for the evaluation of blind source separation.
- [SpeakerBeam](https://github.com/BUTSpeechFIT/speakerbeam) - speakerbeam GIT repository.
- [ConvTasNet](https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/conv_tasnet.py) - convtasnet from Asteroid GIT repository.

## Contact
repository: martin.vratny@tul.cz

article: zbynek.koldovsky@tul.cz

