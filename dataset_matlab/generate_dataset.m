function generate_dataset(nr_mix_samples, datafolder_name, wsj_folder, rng_seed, generate_from_log_path, numTrain, numValidation, numTest, varargin)
    % GENERATE_DATASET Generate multichannel mixed audio dataset for speech separation training
    %
    % This function creates a comprehensive dataset of reverberant audio mixtures suitable for 
    % multichannel speech separation research. It simulates realistic acoustic environments with
    % configurable room acoustics, microphone arrays, and speaker positions.
    % 
    % Depends on:
    % https://github.com/ehabets/RIRu-Generator/tree/master
    % https://www.audiolabs-erlangen.de/fau/professor/habets/software/rir-generator
    %
    % Main Functionality:
    %   1. NEW GENERATION: Creates fresh mixtures from WSJ audio files with random configurations
    %   2. LOG REPRODUCTION: Exactly reproduces datasets from previous generation logs
    %
    % Usage:
    %   % New dataset generation
    %   generate_dataset(nr_mix_samples, datafolder_name, wsj_folder)
    %   generate_dataset(nr_mix_samples, datafolder_name, wsj_folder, rng_seed)
    %   generate_dataset(nr_mix_samples, datafolder_name, wsj_folder, rng_seed, [], numTrain, numValidation, numTest)
    %   
    %   % Reproduction from log file
    %   generate_dataset([], datafolder_name_new, [], [], log_file_path)
    %   generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'NFFT', 1024, 'fs', 8000)
    %
    % Parameters:
    %   nr_mix_samples          - Number of mixed audio samples to generate (use [] for log reproduction)
    %   datafolder_name         - Output directory for generated data files
    %   wsj_folder              - Input directory containing WSJ audio files (ignored for log reproduction)
    %   rng_seed                - (optional) Random seed for reproducibility (default: 42)
    %   generate_from_log_path  - (optional) Path to log file for exact reproduction (default: [])
    %   numTrain                - (optional) Training set size (default: min(1000, total_samples))
    %   numValidation           - (optional) Validation set size (default: min(200, remaining))
    %   numTest                 - (optional) Test set size (default: remaining samples)
    %   varargin                - (optional) Parameter overrides for log reproduction (name-value pairs)
    %
    % Generated Files:
    %   Data Files (in datafolder_name):
    %     - dataN.mat: STFT representations (x=mixture, y=target) for sample N
    %     - sN.wav: Clean target speech (main speaker) for sample N
    %     - yN.wav: Background interference (background speakers) for sample N  
    %     - mixN.wav: Complete mixture (target + background) for sample N
    %   
    %   Metadata Files (in ../info/ directory):
    %     - generation_log.mat: Complete generation parameters and per-sample configurations
    %     - datalisttrain.txt: Training set file indices
    %     - datalistval.txt: Validation set file indices  
    %     - datalisttest.txt: Test set file indices
    %     - generate_dataset.m: Copy of this script for version control
    %
    % Dataset Characteristics:
    %   - 3-microphone linear array (configurable positions)
    %   - Room impulse responses generated with image source method
    %   - Variable SIR (Signal-to-Interference Ratio): 2-10 dB range
    %   - Reverberation time: 0.18s (configurable)
    %   - Sampling rate: 16 kHz
    %   - Recording length: 5 seconds per sample
    %   - STFT parameters: 512-point FFT, 128-sample hop
    %
    % Examples:
    %   % Generate 1500 samples with default splits
    %   generate_dataset(1500, 'D:/dataset/files/', 'D:/wsj_audio/')
    %   
    %   % Generate 100 samples with custom train/val/test splits
    %   generate_dataset(100, 'data/', 'wsj/', 42, [], 60, 20, 20)
    %
    %   % Reproduce existing dataset in new location
    %   generate_dataset([], 'new_location/', [], [], 'old_dataset/info/generation_log.mat')
    %   
    %   % Reproduce with modified STFT parameters
    %   generate_dataset([], 'modified/', [], [], 'log.mat', [], [], [], 'NFFT', 1024, 'FFTSHIFT', 256)
    %
    % Log File Reproduction:
    %   When reproducing from a log file, the function:
    %   - Uses identical speaker selections, positions, and SIR values from original generation
    %   - Applies the same random offsets and sound placement as logged
    %   - Allows parameter overrides via varargin (e.g., STFT settings, room acoustics)
    %   - Creates 'generation_log_regenerated.mat' if parameters are modified
    %   - Generates 'parameter_changes_summary.txt' documenting any changes
    %   - Ensures perfect reproducibility for research validation
    %


    % Input validation
    if nargin < 2
        error('generate_dataset:InsufficientInputs', 'At least 2 inputs required: nr_mix_samples (or []), datafolder_name');
    end
    
    % Check if this is a log file regeneration (5th parameter exists and is not empty)
    is_log_regeneration = (nargin >= 5 && ~isempty(generate_from_log_path));
    
    if ~is_log_regeneration && nargin < 3
        error('generate_dataset:InsufficientInputs', 'For new dataset generation, at least 3 inputs required: nr_mix_samples, datafolder_name, wsj_folder');
    end
    
    % Validate inputs only if not generating from log
    if ~is_log_regeneration
        if isempty(nr_mix_samples) || ~isnumeric(nr_mix_samples) || nr_mix_samples <= 0 || mod(nr_mix_samples, 1) ~= 0
            error('generate_dataset:InvalidSamples', 'nr_mix_samples must be a positive integer');
        end
        
        if ~ischar(datafolder_name) && ~isstring(datafolder_name)
            error('generate_dataset:InvalidDataFolder', 'datafolder_name must be a string or char array');
        end
        
        if ~ischar(wsj_folder) && ~isstring(wsj_folder)
            error('generate_dataset:InvalidWSJFolder', 'wsj_folder must be a string or char array');
        end
        
        if ~exist(wsj_folder, 'dir')
            error('generate_dataset:WSJFolderNotFound', 'WSJ folder does not exist: %s', wsj_folder);
        end
    else
        % For log file generation, only validate datafolder_name
        if ~ischar(datafolder_name) && ~isstring(datafolder_name)
            error('generate_dataset:InvalidDataFolder', 'datafolder_name must be a string or char array');
        end
    end
    
    if nargin < 5
       generate_from_log_path = [];
    end

    if ~isempty(generate_from_log_path)
        % % Handle varargin for parameter overrides
        % fprintf('Debug: Calling generate_dataset_from_log with %d varargin elements\n', length(varargin));
        % if ~isempty(varargin)
        %     fprintf('Debug: varargin contents: ');
        %     for i = 1:length(varargin)
        %         fprintf('%s ', mat2str(varargin{i}));
        %     end
        %     fprintf('\n');
        % end
        generate_dataset_from_log(generate_from_log_path, datafolder_name, varargin{:});
        return; 
    end

    if nargin < 4 || isempty(rng_seed)
        rng_seed = 42; % Default seed
    end
    
    % Set default values for data splits if not provided
    if nargin < 6 || isempty(numTrain)
        numTrain = min(1000, nr_mix_samples); % Default to 1000 or total samples if fewer
    end
    
    if nargin < 7 || isempty(numValidation)
        numValidation = min(200, max(0, nr_mix_samples - numTrain)); % Default to 200 or remaining samples
    end
    
    if nargin < 8 || isempty(numTest)
        numTest = max(0, nr_mix_samples - numTrain - numValidation); % Remaining samples
    end
    
    % Validate that the splits don't exceed total samples
    if numTrain + numValidation + numTest > nr_mix_samples
        error('generate_dataset:InvalidSplits', 'Sum of train (%d) + validation (%d) + test (%d) = %d exceeds total samples (%d)', ...
              numTrain, numValidation, numTest, numTrain + numValidation + numTest, nr_mix_samples);
    end
    
    % Validate split parameters
    if numTrain < 0 || numValidation < 0 || numTest < 0
        error('generate_dataset:NegativeSplits', 'Split values must be non-negative');
    end
    
    fprintf('Data splits: Train=%d, Validation=%d, Test=%d (Total=%d samples)\n', ...
            numTrain, numValidation, numTest, nr_mix_samples);
    
    % Set random seed for reproducibility
    rng(rng_seed);
    

    % --- Parameters to log ---
    params = struct();
    params.rng_seed = rng_seed;
    params.NFFT = 512;
    params.FFTSHIFT = 128;
    params.mics_num = 3;
    params.mics = 1:params.mics_num;
    params.record_len = 5;
    params.max_dist = 1.25;
    params.max_angle = pi/6;
    params.min_dist_bg = 1.5;
    params.num_bg_speakers = 1;
    params.room_size = [5 6 2.5];
    params.ref_point_mic = [2.0 1.25 1.45];
    params.fs = 16000;
    params.mics_pos = 0.01*[-3 0 3; 0 0 3; 3 0 3; -3 0 0; 0 0 0; 3 0 0; -3 0 -3; 0 0 -3; 3 0 -3] + params.ref_point_mic;
    params.datafolder_name = datafolder_name;
    params.wsj_folder = wsj_folder;
    params.nr_mix_samples = nr_mix_samples;

    params.numTrain = numTrain;
    params.numValidation = numValidation;
    params.numTest = numTest;
    % see "function splitLists = create_data_splits(pathToFileDir, numTrain, numValidation, randomSeed)" for more info. 

    params.desired_SNR = +inf; % Changed with removal of noise addition
    % Note: desired_SIR will be generated per mix for variability
    params.desired_SIR_range = [2, 10]; % 2-10 dB range for SIR

    % Parameters for RIR_generator
    params.c = 340;                    % Sound velocity (m/s)
    params.L = params.room_size;       % Room dimensions [x y z] (m)
    params.nsample = 4096;             % Number of samples
    params.beta = 0.18;                % Reverberation time (s)
    params.mtype = 'hypercardioid';    % Type of microphone
    params.order = -1;                 % -1 equals maximum reflection order!
    params.dim = 3;                    % Room dimension
    params.orientation = [pi/2 0];     % Microphone orientation (rad)
    params.hp_filter = 1;              % Enable high-pass filter (1=enable, 0=disable)
    
    % ------------------------------------------------------------------------------------------------------------

    files_in_current_folder = dir(fullfile(wsj_folder, '*'));
    wsj_files = files_in_current_folder(~[files_in_current_folder.isdir]);
    
    % Validate we have enough audio files
    if length(wsj_files) < (params.num_bg_speakers + 1)
        error('generate_dataset:InsufficientFiles', 'Need at least %d audio files, but only %d found in WSJ folder', ...
              params.num_bg_speakers + 1, length(wsj_files));
    end
    
    % Check if all files are unique by their full path (folder + name)
    wsj_fullpaths = arrayfun(@(f) fullfile(f.name), wsj_files, 'UniformOutput', false);
    assert(numel(wsj_fullpaths) == numel(unique(wsj_fullpaths)))

    % Generovani jednotlivych mixu
    % mix_log = struct('main_speaker', {}, 'main_pos', {}, 'bg_speakers', {}, 'bg_positions', {}, 'SIR', {}, 'SNR', {});
    % Mix is preallocated for logging.
    mix_log = struct('main_speaker', {}, ...
                 'main_pos', {}, ...
                 'bg_speakers', {}, ...
                 'bg_positions', {}, ...
                 'num_of_speakers_bg', {}, ...
                 'desired_SIR', {}, ...
                 'desired_SNR', {}, ...
                 'ms_start_pos', {}, ...
                 'ms_offset', {}, ...
                 'bg_start_pos', {}, ...
                 'bg_offset', {},...
                 'noise', {});
    for j=1:nr_mix_samples
       
        % -------------------------------------------------------
        % Hlavni mluvci
        main_speaker_pos = generate_speaker_pos(params.max_dist, params.max_angle, params.ref_point_mic);

        random_index = randi(length(wsj_files)); % Use wsj_files, which holds all file info
        main_speaker_file_info = wsj_files(random_index); % Get the entire struct for the random file
        filename_speaker = fullfile(main_speaker_file_info.folder, main_speaker_file_info.name);
        [main_speaker_full, fs1] = audioread(filename_speaker);
        
        speakers_names = {};
        speakers_names{1} = main_speaker_file_info.name;
    
        % -------------------------------------------------------
        % Priprava matic pro mluvci
        % num_of_speakers_bg = randi([1, params.max_bg_speakers]);
        num_of_speakers_bg = params.num_bg_speakers; % Changed and still logged becouse of easy reverse into variable number of BG speakers.
        bg_speakers = zeros(num_of_speakers_bg, params.fs*params.record_len);
    
        main_speaker = zeros(1, params.fs*params.record_len);
        % main_speaker = add_sound(main_speaker, main_speaker_full, 1);
        [main_speaker,ms_start_pos,ms_offset] = add_sound(main_speaker, main_speaker_full, 1);
        mix_log(j).ms_start_pos = ms_start_pos;
        mix_log(j).ms_offset = ms_offset;

        % -------------------------------------------------------
        % Mluvci v pozadi
        bg_speakers_pos = zeros(num_of_speakers_bg, 3);
        for i=1:num_of_speakers_bg
            % Generate position and check alignment with main speaker
            valid_position = false;
            max_attempts = 50; % Prevent infinite loop
            attempt_count = 0;
            
            while ~valid_position && attempt_count < max_attempts
                attempt_count = attempt_count + 1;
                bg_speakers_pos(i, :) = generate_bg_pos(params.min_dist_bg, params.ref_point_mic, params.room_size);
                
                % Check if this background speaker is aligned with the main speaker
                if are_speakers_aligned(params.ref_point_mic, main_speaker_pos, bg_speakers_pos(i, :), 5)
                    % Speakers are aligned (within 5 degrees) - try again
                    continue;
                end
                
                % Check alignment with any previously placed background speakers
                aligned_with_previous = false;
                for k = 1:(i-1)
                    if are_speakers_aligned(params.ref_point_mic, bg_speakers_pos(k, :), bg_speakers_pos(i, :), 5)
                        aligned_with_previous = true;
                        break;
                    end
                end
                
                % If not aligned with any previous speaker, position is valid
                if ~aligned_with_previous
                    valid_position = true;
                end
            end
            
            % If we couldn't find a valid position after max attempts, warn but continue
            if ~valid_position
                error('Could not find non-aligned position for background speaker %d after %d attempts', i, max_attempts);
            end
            % --- filtering of files ---
            % Exclude file that contains the main speaker's file name (01aa010d.wav) -> removes single file selected as main speaker
            filtered_wsj_files = wsj_files(~contains({wsj_files.name}, main_speaker_file_info.name)); % First check for main speaker
            
            % Exclude all files from a main speaker (01aa010d.wav) -> removes all files from speaker '01a'.
            %%%
            % filtered_wsj_files = wsj_files(~contains({wsj_files.name}, main_speaker_file_info.name(1:3)));
            %%%

            % Exclude already selected background speakers
            already_selected = ismember({filtered_wsj_files.name}, speakers_names); % Second check for rest of the BG speakers
            filtered_wsj_files = filtered_wsj_files(~already_selected); % only unique speakers.
            
            % Check if we have enough speakers remaining
            remaining_bg_speakers_needed = num_of_speakers_bg - (i - 1);
            if isempty(filtered_wsj_files) || (length(filtered_wsj_files) < remaining_bg_speakers_needed)
                error('No background speakers available that satisfy your request. Need %d more speakers, but only %d available.', ...
                      remaining_bg_speakers_needed, length(filtered_wsj_files));
            end
            % --- filtering of files end ---
            random_index = randi(length(filtered_wsj_files)); % Use the filtered list files
            random_bg_speaker_file_info = filtered_wsj_files(random_index);
            random_bg_speaker_name = random_bg_speaker_file_info.name; % Get just the name for comparison
 
            speakers_names{i+1} = random_bg_speaker_name; 
            filename_bg_speaker = fullfile(random_bg_speaker_file_info.folder, random_bg_speaker_file_info.name);

            [bg_speaker, fs2] = audioread(filename_bg_speaker);
            % bg_speakers = add_sound(bg_speakers, bg_speaker, i);
            [bg_speakers,bg_start_pos,bg_offset] = add_sound(bg_speakers, bg_speaker, i);
            mix_log(j).bg_start_pos(i) = bg_start_pos; 
            mix_log(j).bg_offset(i) = bg_offset;

            if fs1 ~= params.fs || fs2 ~= params.fs
                error('Sample rate mismatch between audio files and RIR parameters.');
            end
    
        end
            
        % Generate RIR (Room Impulse Response)
        h_bg = zeros(params.mics_num, params.nsample, num_of_speakers_bg);
        h_main = zeros(params.mics_num, params.nsample);
        for m=1:params.mics_num
            for n=1:num_of_speakers_bg
                h_bg(m, :, n) = generate_rir(params.mics_pos(m, :), bg_speakers_pos(n, :), params);
            end
            h_main(m, :) = generate_rir(params.mics_pos(m, :), main_speaker_pos, params);
        end

        % % Visualize impulse response for the first microphone and main speaker
        % if j == 1 % Only for the first mix to avoid too many plots
        %     % Combined visualization of impulse response and room setup
        %     rir_vizual(j,ref_point_mic, main_speaker_pos, room_size, mics_pos, bg_speakers_pos);
        % end

        % % Viz EVERY mix
        % rir_vizual(j,ref_point_mic, main_speaker_pos, room_size, mics_pos, bg_speakers_pos, speakers_names);


        % Konvoluce
        bg_mixed_conv = zeros(params.mics_num,length(main_speaker));
        main_conv = zeros(params.mics_num,length(main_speaker));
        for m = 1:params.mics_num
            for n = 1:num_of_speakers_bg
                conv_speaker = filter(h_bg(m,:,n), 1, bg_speakers(n,:));
                bg_mixed_conv(m, :) = bg_mixed_conv(m, :) + conv_speaker;
            end
            main_conv(m, :) = filter(h_main(m,:), 1, main_speaker);
        end
    
        % Noise addition: (SNR setting) - Removed for this version
        signal_power = mean(main_conv.^2, "all"); % needed for SIR
        
        % SIR adjustment (background speakers vs main speaker level):
        % Generate SIR per mix for variability
        desired_SIR = rand() * (params.desired_SIR_range(2) - params.desired_SIR_range(1)) + params.desired_SIR_range(1);
        
        sir_linear = 10^(desired_SIR / 10);
        int_power_old = mean(bg_mixed_conv.^2, "all");
        scale_factor = sqrt(signal_power / (int_power_old * sir_linear));
        bg_mixed_conv = scale_factor * bg_mixed_conv;
    
        % Normalization
        mix_conv = bg_mixed_conv + main_conv;
        max_abs = max(abs([mix_conv; bg_mixed_conv; main_conv]),[],"all");
    
        bg_mixed_conv = bg_mixed_conv / max_abs;
        main_conv = main_conv / max_abs;
        mix_conv = mix_conv / max_abs;

        % Check for zeroed mixtures
        if(or(sum(mix_conv,"all")==0,sum(isnan(mix_conv),"all")~=0))
            error('ERROR: Invalid mixture');
        end    

        % STFT
        x_stft = stftm(mix_conv, params.NFFT, params.FFTSHIFT, params.NFFT, hamming(params.NFFT));
        x_stft = x_stft(params.mics,:,1:params.NFFT/2+1);
        y_stft = stftm(main_conv,params.NFFT, params.FFTSHIFT, params.NFFT, hamming(params.NFFT));
        y_stft = y_stft(params.mics,:,1:params.NFFT/2+1);

        % Storage
        fprintf('data idx: %d/%d\n', j, nr_mix_samples)

        % Check if datafolder_name exists, create if not
        if ~exist(datafolder_name, 'dir')
           mkdir(datafolder_name);
        end
        x = single(x_stft);
        y = single(y_stft);
        save(fullfile(datafolder_name, ['data', num2str(j-1)]), 'x', 'y');
        
        % % Determine the output format -> Single -> and specify for audiowrite by BitsPerSample
        % main_conv_single = single(main_conv);
        % bg_mixed_conv_single = single(bg_mixed_conv);
        % mix_conv_single = single(mix_conv);

        audiowrite(fullfile(datafolder_name, ['s', num2str(j-1), '.wav']), main_conv', params.fs);
        audiowrite(fullfile(datafolder_name, ['y', num2str(j-1), '.wav']), bg_mixed_conv', params.fs);
        audiowrite(fullfile(datafolder_name, ['mix', num2str(j-1), '.wav']), mix_conv', params.fs);
        
        % log the relevant info for each mix:
        mix_log(j).main_speaker = main_speaker_file_info.name;
        mix_log(j).main_pos = main_speaker_pos;
        mix_log(j).bg_speakers = speakers_names(2:end);
        mix_log(j).bg_positions = bg_speakers_pos;
        mix_log(j).num_of_speakers_bg = num_of_speakers_bg;
        mix_log(j).desired_SIR = desired_SIR; % Use the per-mix generated SIR
        mix_log(j).desired_SNR = params.desired_SNR;
    end

    % Create info directory at the same level as the data folder
    if datafolder_name(end) == filesep
        parent_dir = fileparts(datafolder_name(1:end-1));
    else    
        parent_dir = fileparts(datafolder_name);
    end
    
    % If datafolder_name is already a subdirectory, go up one more level
    if strcmp(fileparts(parent_dir), parent_dir) % Check if we're at root
        mod_datafolder_name = fullfile(parent_dir, 'info');
    else
        mod_datafolder_name = fullfile(fileparts(parent_dir), 'info');
    end

    if ~exist(mod_datafolder_name, 'dir')
       mkdir(mod_datafolder_name);
    end

    log_generation_metadata(fullfile(mod_datafolder_name, 'generation_log.mat'), params, mix_log);
    save_current_script_for_versioning(fullfile(mod_datafolder_name));

    % Create and save lists for Python processing
    my_splits = create_data_splits(datafolder_name,params.numTrain,params.numValidation,params.rng_seed);
  
    save_list_to_txt(my_splits.train, fullfile(mod_datafolder_name,'datalisttrain.txt'));
    save_list_to_txt(my_splits.validation, fullfile(mod_datafolder_name,'datalistval.txt'));
    save_list_to_txt(my_splits.test, fullfile(mod_datafolder_name,'datalisttest.txt'));
end

function generate_dataset_from_log(log_file, datafolder_name_new, varargin)
% GENERATE_DATASET_FROM_LOG Reproduce multichannel audio dataset from generation log
%
% This function provides exact reproduction of previously generated datasets by reading
% all generation parameters and per-sample configurations from a log file. It ensures
% perfect reproducibility for research validation while allowing selective parameter
% modifications for ablation studies.
%
% Key Features:
%   - Exact reproduction of speaker selections and acoustic configurations
%   - Preservation of random offsets and sound placement timing
%   - Optional parameter overrides via name-value pairs
%   - Automatic change tracking and documentation
%   - Support for output location changes
%
% Usage:
%   generate_dataset_from_log(log_file, datafolder_name_new)
%   generate_dataset_from_log(log_file, datafolder_name_new, 'param1', value1, 'param2', value2, ...)
%
% Parameters:
%   log_file            - Path to generation_log.mat file from previous generation
%   datafolder_name_new - Output directory for reproduced dataset (can differ from original)
%   varargin           - Optional parameter overrides as name-value pairs
%
% Reproduction Process:
%   1. Loads complete generation state from log file (params + mix_log)
%   2. Validates that all required WSJ audio files are still accessible
%   3. Recreates each mixture using exact same:
%      - Speaker file selections (main and background speakers)
%      - Spatial positions for all sources
%      - Signal-to-Interference Ratio (SIR) values
%      - Audio segment timing and offsets
%   4. Applies room impulse responses with identical parameters
%   5. Generates same file outputs as original generation
%
% Parameter Override Examples:
%   % Change STFT parameters while keeping everything else identical
%   generate_dataset_from_log('log.mat', 'new_output/', 'NFFT', 1024, 'FFTSHIFT', 256)
%   
%   % Modify room acoustics for ablation study
%   generate_dataset_from_log('log.mat', 'reverb_study/', 'beta', 0.3, 'room_size', [6 7 3])
%   
%   % Change sampling rate and corresponding parameters
%   generate_dataset_from_log('log.mat', 'fs8k/', 'fs', 8000, 'nsample', 2048)
%
% Generated Files:
%   Same structure as original generation:
%   - dataN.mat: STFT representations (with any parameter modifications applied)
%   - sN.wav, yN.wav, mixN.wav: Time-domain audio files  
%   - Train/validation/test split files
%   - generation_log_regenerated.mat: Updated log if parameters were changed
%   - parameter_changes_summary.txt: Human-readable change documentation
%
% Change Tracking:
%   When parameters are modified, the function automatically:
%   - Creates 'generation_log_regenerated.mat' with updated parameters
%   - Saves 'parameter_changes_summary.txt' documenting all changes
%   - Preserves original log file path and regeneration timestamp
%   - Maintains full traceability for research reproducibility
%
% Limitations:
%   - Only acoustic/processing parameters can be overridden
%   - Mix configurations (speakers, positions, SIR values) remain unchanged
%   - All original WSJ audio files must be accessible at logged paths
%   - varargin must contain valid parameter names from original generation
%
% Note: This function is automatically called by generate_dataset() when a log file
%       path is provided as the 5th parameter. Direct calls are also supported.

    % Input validation
    if nargin < 2
        error('generate_dataset_from_log:InsufficientInputs', 'Both log_file and datafolder_name_new are required');
    end
    
    % Validate varargin has even number of elements (name-value pairs)
    fprintf('Debug: generate_dataset_from_log received %d varargin elements\n', length(varargin));
    if ~isempty(varargin)
        fprintf('Debug: varargin contents in generate_dataset_from_log: ');
        for i = 1:length(varargin)
            fprintf('%s ', mat2str(varargin{i}));
        end
        fprintf('\n');
    end
    if mod(length(varargin), 2) ~= 0
        error('generate_dataset_from_log:InvalidVarargin', 'varargin must contain name-value pairs (even number of elements)');
    end
    
    if ~exist(log_file, 'file')
        error('generate_dataset_from_log:LogFileNotFound', 'Log file does not exist: %s', log_file);
    end

    % Load all params and mix_log from the log file for exact replication
    try
        loaded = load(log_file);
        if ~isfield(loaded, 'params') || ~isfield(loaded, 'mix_log')
            error('generate_dataset_from_log:InvalidLogFile', 'Log file must contain params and mix_log variables');
        end
        params = loaded.params;
        mix_log = loaded.mix_log;
    catch ME
        error('generate_dataset_from_log:LoadError', 'Failed to load log file: %s', ME.message);
    end
    
    % Track parameter overrides for logging
    param_overrides = struct();
    original_params = params; % Keep a copy of original parameters
    has_param_changes = false;
    
    % Override parameters with varargin name-value pairs
    for i = 1:2:length(varargin)
        param_name = varargin{i};
        param_value = varargin{i+1};
        
        if ~ischar(param_name) && ~isstring(param_name)
            error('generate_dataset_from_log:InvalidParamName', 'Parameter names must be strings or char arrays');
        end
        
        if isfield(params, param_name)
            fprintf('Overriding parameter %s: %s -> %s\n', param_name, mat2str(params.(param_name)), mat2str(param_value));
            param_overrides.(param_name) = struct('original', params.(param_name), 'new', param_value);
            params.(param_name) = param_value;
            has_param_changes = true;
        else
            warning('generate_dataset_from_log:UnknownParam', 'Parameter "%s" not found in original params. Adding as new parameter.', param_name);
            param_overrides.(param_name) = struct('original', 'N/A (new parameter)', 'new', param_value);
            params.(param_name) = param_value;
            has_param_changes = true;
        end
    end
    
    % Extract parameters from log
    NFFT = params.NFFT;
    FFTSHIFT = params.FFTSHIFT;
    mics_num = params.mics_num;
    mics = params.mics;
    record_len = params.record_len;
    fs = params.fs;
    mics_pos = params.mics_pos;
    ref_point_mic = params.ref_point_mic;
    room_size = params.room_size;

    datafolder_name = params.datafolder_name; % from log
    if ~strcmp(datafolder_name, datafolder_name_new) % if you want to save to different location then change the directory from log.
        datafolder_name = datafolder_name_new;
    end
    wsj_folder = params.wsj_folder;
    nr_mix_samples = params.nr_mix_samples;
    numTrain = params.numTrain;
    numValidation = params.numValidation;
    % % Extract numTest if available (for completeness, though not strictly needed)
    % if isfield(params, 'numTest')
    %     numTest = params.numTest;
    % else
    %     numTest = nr_mix_samples - numTrain - numValidation;
    % end
    rng_seed = params.rng_seed;
    
    % Validate WSJ folder from log still exists
    if ~exist(wsj_folder, 'dir')
        error('generate_dataset_from_log:WSJFolderNotFound', 'WSJ folder from log does not exist: %s', wsj_folder);
    end

    files_in_current_folder = dir(fullfile(wsj_folder, '*'));
    wsj_files = files_in_current_folder(~[files_in_current_folder.isdir]);
    wsj_names = {wsj_files.name};

    for j = 1:nr_mix_samples
        % Main speaker
        main_speaker_name = mix_log(j).main_speaker;
        main_speaker_idx = find(strcmp(wsj_names, main_speaker_name), 1);
        if isempty(main_speaker_idx)
            error('generate_dataset_from_log:MainSpeakerNotFound', 'Main speaker file not found: %s', main_speaker_name);
        end
        main_speaker_file_info = wsj_files(main_speaker_idx);
        filename_speaker = fullfile(main_speaker_file_info.folder, main_speaker_file_info.name);
        [main_speaker_full, fs1] = audioread(filename_speaker);
        main_speaker_pos = mix_log(j).main_pos;

        num_of_speakers_bg = length(mix_log(j).bg_speakers); % is in logs explicitly 
        
        bg_speakers = zeros(num_of_speakers_bg, fs*record_len);
        main_speaker = zeros(1, fs*record_len);
        main_speaker = add_sound_log(main_speaker, main_speaker_full, 1, mix_log(j).ms_start_pos, mix_log(j).ms_offset);

        bg_speakers_pos = mix_log(j).bg_positions;
        % Background speakers
        for i = 1:num_of_speakers_bg
            bg_name = mix_log(j).bg_speakers{i};
            bg_idx = find(strcmp(wsj_names, bg_name), 1);
            if isempty(bg_idx)
                error('generate_dataset_from_log:BackgroundSpeakerNotFound', 'Background speaker file not found: %s', bg_name);
            end
            bg_file_info = wsj_files(bg_idx);
            filename_bg_speaker = fullfile(bg_file_info.folder, bg_file_info.name);
            [bg_speaker, fs2] = audioread(filename_bg_speaker);
            bg_speakers = add_sound_log(bg_speakers, bg_speaker, i, mix_log(j).bg_start_pos(i), mix_log(j).bg_offset(i));
            if fs1 ~= fs || fs2 ~= fs
                error('Sample rate mismatch between audio files and RIR parameters.');
            end
        end
        % desired_SIR = mix_log(j).desired_SIR;
        % desired_SNR = mix_log(j).desired_SNR;

        h_bg = zeros(mics_num, params.nsample, num_of_speakers_bg);
        h_main = zeros(mics_num, params.nsample);
        for m=1:mics_num
            for n=1:num_of_speakers_bg
                h_bg(m, :, n) = generate_rir(mics_pos(m, :), bg_speakers_pos(n, :), params);
            end
            h_main(m, :) = generate_rir(mics_pos(m, :), main_speaker_pos, params);
        end

        % Visualize impulse response and room setup (from log reproduction)
        % Create speakers_names array for visualization
        speakers_names = cell(1, num_of_speakers_bg + 1);
        speakers_names{1} = main_speaker_name; % Main speaker first
        for i = 1:num_of_speakers_bg
            speakers_names{i+1} = mix_log(j).bg_speakers{i}; % Background speakers
        end
        
        bg_mixed_conv = zeros(mics_num,length(main_speaker));
        main_conv = zeros(mics_num,length(main_speaker));
        for m = 1:mics_num
            for n = 1:num_of_speakers_bg
                conv_speaker = filter(h_bg(m,:,n), 1, bg_speakers(n,:));
                bg_mixed_conv(m, :) = bg_mixed_conv(m, :) + conv_speaker;
            end
            main_conv(m, :) = filter(h_main(m,:), 1, main_speaker);
        end

        signal_power = mean(main_conv.^2, "all");
        % snr_linear = 10^(desired_SNR / 10);
        % noise_power = signal_power / snr_linear;
        % noise = sqrt(noise_power) * randn(mics_num, size(bg_mixed_conv, 2));
        % % noise = mix_log(j).noise;

        % SIR reproduction logic
        % Check if SIR parameters were changed
        sir_params_changed = isfield(param_overrides, 'desired_SIR_range') || isfield(param_overrides, 'desired_SIR');
        
        if ~sir_params_changed
            % Exact reproduction - use stored SIR value
            desired_SIR = mix_log(j).desired_SIR;
        else
            % SIR parameters changed - generate new SIR using provided parameters
            original_SIR = mix_log(j).desired_SIR; % Store original for logging
            new_desired_SIR = params.desired_SIR_range(1) + (params.desired_SIR_range(2) - params.desired_SIR_range(1)) * rand();
            desired_SIR = new_desired_SIR;
            
            % IMPORTANT: Update mix_log with new SIR data for proper log saving and possible future regeneration
            mix_log(j).desired_SIR = new_desired_SIR;
            
            fprintf('Info: Mix %d - Generated new SIR %.2f dB (was %.2f dB) due to parameter override.\n', ...
                    j, new_desired_SIR, original_SIR);
        end

        sir_linear = 10^(desired_SIR / 10);
        int_power_old = mean(bg_mixed_conv.^2, "all");
        scale_factor = sqrt(signal_power / (int_power_old * sir_linear));
        bg_mixed_conv = scale_factor * bg_mixed_conv;

        % bg_mixed_conv = bg_mixed_conv + noise; % change with removal of noise

        mix_conv = bg_mixed_conv + main_conv;

        max_abs = max(abs([mix_conv; bg_mixed_conv; main_conv]),[],'all');

        bg_mixed_conv = bg_mixed_conv / max_abs;
        main_conv = main_conv / max_abs;
        mix_conv = mix_conv / max_abs;

        if(or(sum(mix_conv,"all")==0,sum(isnan(mix_conv),"all")~=0))
            error('ERROR: Invalid mixture');
        end

        % %%%% Show visualization for every mix during log reproduction (after convolution)
        % fig_handle = rir_vizual(j, ref_point_mic, main_speaker_pos, room_size, mics_pos, bg_speakers_pos, speakers_names, params, main_conv, bg_mixed_conv);
        % 
        % % Create rir_vizual directory if it doesn't exist
        % if datafolder_name(end) == filesep || datafolder_name(end) == '/'
        %     rir_vizual_dir = fullfile(fileparts(datafolder_name(1:end-1)), 'rir_vizual');
        % else    
        %     rir_vizual_dir = fullfile(fileparts(datafolder_name), 'rir_vizual');
        % end
        % 
        % if ~exist(rir_vizual_dir, 'dir')
        %     mkdir(rir_vizual_dir);
        % end
        % 
        % % Save the figure
        % figure_filename = fullfile(rir_vizual_dir, sprintf('mix%d.png', j-1));
        % saveas(fig_handle, figure_filename);
        % figure_filename = fullfile(rir_vizual_dir, sprintf('mix%d.fig', j-1));
        % saveas(fig_handle, figure_filename);
        % 
        % % Close the figure to free memory
        % close(fig_handle);

        x_stft = stftm(mix_conv, NFFT, FFTSHIFT, NFFT, hamming(NFFT));
        x_stft = x_stft(mics,:,1:NFFT/2+1);
        y_stft = stftm(main_conv,NFFT, FFTSHIFT, NFFT, hamming(NFFT));
        y_stft = y_stft(mics,:,1:NFFT/2+1);
        fprintf('data idx: %d/%d\n', j, nr_mix_samples)

        if ~exist(datafolder_name, 'dir')
           mkdir(datafolder_name);
        end

        x = single(x_stft);
        y = single(y_stft);

        save(fullfile(datafolder_name, ['data', num2str(j-1)]), 'x', 'y');

        audiowrite(fullfile(datafolder_name, ['s', num2str(j-1), '.wav']), main_conv', fs);
        audiowrite(fullfile(datafolder_name, ['y', num2str(j-1), '.wav']), bg_mixed_conv', fs);
        audiowrite(fullfile(datafolder_name, ['mix', num2str(j-1), '.wav']), mix_conv', fs);
    end
    % Create info directory at the same level as the data folder
    if datafolder_name(end) == filesep
        parent_dir = fileparts(datafolder_name(1:end-1));
    else    
        parent_dir = fileparts(datafolder_name);
    end
    
    % If datafolder_name is already a subdirectory, go up one more level
    if strcmp(fileparts(parent_dir), parent_dir) % Check if we're at root
        mod_datafolder_name = fullfile(parent_dir, 'info');
    else
        mod_datafolder_name = fullfile(fileparts(parent_dir), 'info');
    end

    if ~exist(mod_datafolder_name, 'dir')
       mkdir(mod_datafolder_name);
    end
    save_current_script_for_versioning(fullfile(mod_datafolder_name));

    % Save new log if parameters were changed
    if has_param_changes
        % Create regeneration metadata
        regeneration_metadata = struct();
        regeneration_metadata.original_log_file = log_file;
        regeneration_metadata.regeneration_timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        regeneration_metadata.original_datafolder = original_params.datafolder_name;
        regeneration_metadata.new_datafolder = datafolder_name;
        regeneration_metadata.param_overrides = param_overrides;
        regeneration_metadata.has_param_changes = has_param_changes;
        
        % Save the new log with updated parameters and regeneration info
        log_filename = fullfile(mod_datafolder_name, 'generation_log_regenerated.mat');
        fprintf('Saving regenerated log with parameter changes to: %s\n', log_filename);
        save(log_filename, 'params', 'mix_log', 'regeneration_metadata', 'original_params', '-v7.3');
        
        % Also save a summary of changes as a text file for easy review
        changes_summary_file = fullfile(mod_datafolder_name, 'parameter_changes_summary.txt');
        write_parameter_changes_summary(changes_summary_file, param_overrides, log_file, datafolder_name);
    else
        fprintf('No parameter changes detected. Regenerated dataset uses identical parameters from original log.\n');
    end

    my_splits = create_data_splits(datafolder_name,numTrain,numValidation,rng_seed);

    save_list_to_txt(my_splits.train, fullfile(mod_datafolder_name,'datalisttrain.txt'));
    save_list_to_txt(my_splits.validation, fullfile(mod_datafolder_name,'datalistval.txt'));
    save_list_to_txt(my_splits.test, fullfile(mod_datafolder_name,'datalisttest.txt'));
end

function log_generation_metadata(logfile, params, mix_log)
    % Save all relevant parameters and per-mix info to a .mat file for reproducibility
    % When saving 1500 files the log file will have around 2,8 GB size.
    % Main chunk is from storing the noise vector. That is used for identical regeneration of data.
    save(logfile, 'params', 'mix_log', "-v7.3");
end

function write_parameter_changes_summary(filename, param_overrides, original_log_file, new_datafolder)
% WRITE_PARAMETER_CHANGES_SUMMARY Creates a human-readable summary of parameter changes
%
% This function writes a text file summarizing all parameter changes made during
% dataset regeneration from a log file. This provides an easily readable record
% of what was modified compared to the original generation.
%
% Parameters:
%   filename         - (char array) Full path to the output text file
%   param_overrides  - (struct) Structure containing parameter changes with
%                      'original' and 'new' fields for each changed parameter
%   original_log_file - (char array) Path to the original log file used for regeneration
%   new_datafolder   - (char array) Path to the new data folder where regenerated data is saved
%
% The output file contains:
%   - Header with regeneration information
%   - List of all parameter changes with original and new values
%   - Timestamp of when the summary was created

    fid = fopen(filename, 'w');
    if fid == -1
        error('write_parameter_changes_summary:FileOpenError', 'Could not open file %s for writing.', filename);
    end
    
    try
        % Write header
        fprintf(fid, '=============================================================\n');
        fprintf(fid, 'DATASET REGENERATION - PARAMETER CHANGES SUMMARY\n');
        fprintf(fid, '=============================================================\n\n');
        
        % Write metadata
        fprintf(fid, 'Regeneration Timestamp: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        fprintf(fid, 'Original Log File: %s\n', original_log_file);
        fprintf(fid, 'New Data Folder: %s\n\n', new_datafolder);
        
        % Write parameter changes
        param_names = fieldnames(param_overrides);
        fprintf(fid, 'PARAMETER CHANGES (%d total):\n', length(param_names));
        fprintf(fid, '-------------------------------------------------------------\n\n');
        
        for i = 1:length(param_names)
            param_name = param_names{i};
            override_info = param_overrides.(param_name);
            
            fprintf(fid, 'Parameter: %s\n', param_name);
            fprintf(fid, '  Original: %s\n', format_value_for_text(override_info.original));
            fprintf(fid, '  New:      %s\n', format_value_for_text(override_info.new));
            fprintf(fid, '\n');
        end
        
        % Write footer
        fprintf(fid, '-------------------------------------------------------------\n');
        fprintf(fid, 'End of Parameter Changes Summary\n');
        fprintf(fid, 'Generated by generate_dataset.m\n');
        
    catch ME
        fclose(fid);
        error('write_parameter_changes_summary:WriteError', 'Error writing to file: %s', ME.message);
    end
    
    fclose(fid);
    fprintf('Parameter changes summary saved to: %s\n', filename);
end

function str = format_value_for_text(value)
% FORMAT_VALUE_FOR_TEXT Converts various MATLAB data types to readable text
%
% This helper function formats different types of MATLAB variables into
% human-readable strings for text file output.

    if ischar(value) || isstring(value)
        str = char(value);
    elseif isnumeric(value)
        if numel(value) == 1
            str = num2str(value);
        elseif numel(value) <= 10  % Show small arrays inline
            str = mat2str(value);
        else
            str = sprintf('[%dx%d %s array]', size(value, 1), size(value, 2), class(value));
        end
    elseif islogical(value)
        if value
            str = 'true';
        else
            str = 'false';
        end
    elseif iscell(value)
        str = sprintf('{%dx%d cell array}', size(value, 1), size(value, 2));
    elseif isstruct(value)
        str = sprintf('[%dx%d struct with %d fields]', size(value, 1), size(value, 2), length(fieldnames(value)));
    else
        str = sprintf('[%s object]', class(value));
    end
end

function [sounds,start_pos,offset] = add_sound(sounds, new_sound, index)
    % Add sound to sound matrix - adjust length and random placement
    len_sound = length(sounds);
    
    if length(new_sound) >= len_sound
        offset = NaN; % because of logging and return values (see "add_sound_log")
        start_pos = randi([1, length(new_sound) - len_sound + 1]);
        sounds(index, :) = new_sound(start_pos:start_pos+len_sound-1);
        
    else
        start_pos = NaN; % because of logging and return values (see "add_sound_log")
        len_new = length(new_sound);
        offset = randi([0, len_sound - len_new]);
        
        new_segment = zeros(1, len_sound);
        new_segment(offset+1:offset+len_new) = new_sound;
        
        sounds(index, :) = new_segment;
    end
end

function sounds = add_sound_log(sounds, new_sound, index, start_pos, offset)
    % Copy of add_sound for regenerating dataset from log
    % This is done because of rng functionality and my limited knowledge
    % but it should work and create identical dataset based on log file
    % created in original "generate_dataset" func.

    % Add sound to sound matrix - adjust length and random placement
    len_sound = length(sounds);
    
    if length(new_sound) >= len_sound
        % start_pos = randi([1, length(new_sound) - len_sound + 1]);
        sounds(index, :) = new_sound(start_pos:start_pos+len_sound-1);
        
    else
        len_new = length(new_sound);
        % offset = randi([0, len_sound - len_new]);

        new_segment = zeros(1, len_sound);
        new_segment(offset+1:offset+len_new) = new_sound;
        
        sounds(index, :) = new_segment;
    end
end


function h = generate_rir(r,s,params)
    % % Original - without log
    % c = 340;                    % Sound velocity (m/s)
    % fs = 16000;                 % Sample frequency (samples/s)
    % L = room_size;              % Room dimensions [x y z] (m)
    % nsample = 4096;             % Number of samples
    % beta = 0.18;                 % Reverberation time (s)
    % mtype = 'hypercardioid';    % Type of microphone
    % order = -1;                 % -1 equals maximum reflection order!
    % dim = 3;                    % Room dimension
    % orientation = [pi/2 0];     % Microphone orientation (rad)
    % hp_filter = 1;              % Disable high-pass filter

    % h = rir_generator(c, fs, r, s, L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    % h = rir_generator(params.c, params.fs, r, s, params.L, params.beta, params.nsample, params.mtype, params.order, params.dim, params.orientation, params.hp_filter);
    %%% Mistake fix for duplication of parameters.
    h = rir_generator(params.c, params.fs, r, s, params.room_size, params.beta, params.nsample, params.mtype, params.order, params.dim, params.orientation, params.hp_filter);
end

function pos = generate_speaker_pos(max_dist, max_angle, ref_point_mic)
    valueY = (0.1 + 0.9 * rand()) * max_dist;
    limit1 = sqrt((max_dist^2) - (valueY^2)); % maximum distance constraint
    limit2 = tan(max_angle) * valueY; % angle constraint
    maxX = min(limit1, limit2);
    valueX = rand() * (2*maxX) - maxX; % possibility on both sides
    valueZ = 1.3 + (1.85-1.3)*rand(); % speaker height

    pos = round([valueX + ref_point_mic(1), valueY + ref_point_mic(2), valueZ], 2);
end

function pos = generate_bg_pos(min_dist, ref_point_mic, room_size)
    found_pos = false;

    % always be in front of ref_point
    min_y_for_bg = ref_point_mic(2);
    max_y_for_bg = room_size(2);
    if min_y_for_bg >= max_y_for_bg
        error('generate_bg_pos:InvalidRoomSetup', 'Microphone Y-position (%f) is at or beyond the room''s max Y-dimension (%f). Cannot place background speaker in front.', ref_point_mic(2), room_size(2));
    end
    
    while ~found_pos % potentially infinite loop
        valueX = rand() * room_size(1);
        % valueY = rand() * room_size(2);

        % always be in front of ref_point
        valueY = min_y_for_bg + (max_y_for_bg - min_y_for_bg) * rand;

        dist = sqrt((valueX - ref_point_mic(1))^2 + (valueY - ref_point_mic(2))^2);
        if dist > min_dist
            found_pos = true;
        end
    end
    valueZ = 1.3 + (1.85-1.3)*rand();

    pos = round([valueX, valueY, valueZ],2);
end


function fig_handle = rir_vizual(id,ref_point_mic, main_speaker_pos, room_size, mics_pos, bg_speakers_pos, speakers_filenames, params, main_conv, bg_mixed_conv) %#ok<DEFNU>
    % This function creates a combined visualization of room impulse response and 3D room setup
    %
    % Parameters:
    %   ref_point_mic - reference microphone position [x y z] in meters
    %   main_speaker_pos - main speaker position [x y z] in meters
    %   room_size - dimensions of the room [x y z] in meters
    %   mics_pos - positions of all microphones [N x 3] in meters
    %   bg_speakers_pos - positions of background speakers [M x 3] in meters
    %   speakers_filenames - cell array of filenames, first is main, rest are background
    %   params - parameters struct containing acoustic settings
    %   main_conv - convolved main speaker audio [channels x samples]
    %   bg_mixed_conv - convolved background speaker audio [channels x samples]
    %
    % Returns:
    %   fig_handle - handle to the created figure
    
    % Create a new figure with 6 subplots (2x3 layout)
    fig_handle = figure('Name',sprintf("Mix %1d",id),'Position', [100, 100, 1500, 700], Visible='off');
    
    % Generate impulse responses for all speakers
    h_main = generate_rir(ref_point_mic, main_speaker_pos, params);
    fs = 16000; % Sample rate
    
    % Generate background speaker impulse responses
    h_bg = cell(size(bg_speakers_pos, 1), 1);
    for i = 1:size(bg_speakers_pos, 1)
        h_bg{i} = generate_rir(ref_point_mic, bg_speakers_pos(i,:), params);
    end
    
    % 1. Plot time domain impulse responses comparison (Top-Left)
    subplot(2, 3, 1);
    % t = (0:length(h_main)-1) * 1000 / fs; % Convert to milliseconds
    
    % Plot main speaker (blue)
    plot(h_main, 'b-', 'DisplayName', 'Main Speaker');
    hold on;
    
    % Plot background speakers (different colors)
    colors = {'r-', 'm-', 'c-', 'g-', 'k-'}; % Red, Magenta, Cyan, Green, Black
    for i = 1:length(h_bg)
        color_idx = mod(i-1, length(colors)) + 1;
        if length(speakers_filenames) > i % Has background speaker names
            [~, name, ~] = fileparts(speakers_filenames{i+1}); % Remove extension for cleaner display
            display_name = sprintf('BG Speaker %d (%s)', i, name);
        else
            display_name = sprintf('BG Speaker %d', i);
        end
        plot(h_bg{i}, colors{color_idx}, 'LineWidth', 1, 'DisplayName', display_name);
    end
    
    hold off;
    title('Room Impulse Response Comparison (Time Domain)');
    xlabel('Time (ms)');
    ylabel('Amplitude');
    legend('show', 'Location', 'best');
    grid on;
    
    % 2. Plot frequency response (Top-Middle)
    subplot(2, 3, 2);
    [H, f] = freqz(h_main, 1, 1024, fs);
    semilogx(f, 20*log10(abs(H)));
    title('Room Frequency Response (Main Speaker)');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    xlim([20, fs/2]); % Limit to audible range
    
    % 3. Plot spectrogram for main speaker (Top-Right)
    subplot(2, 3, 3);
    spectrogram(main_conv(1,:),512,128,512,'y');
    title('Main Speaker Spectrogram');
    % xlabel('Time (ms)');
    % ylabel('Frequency (kHz)');
    colorbar;
    
    % 4. Plot 3D room visualization (Bottom-Left)
    subplot(2, 3, 4);
    
    % Plot room boundaries
    plot3([0, room_size(1), room_size(1), 0, 0], [0, 0, room_size(2), room_size(2), 0], [0, 0, 0, 0, 0], 'k-');
    hold on;
    plot3([0, room_size(1), room_size(1), 0, 0], [0, 0, room_size(2), room_size(2), 0], [room_size(3), room_size(3), room_size(3), room_size(3), room_size(3)], 'k-');
    plot3([0, 0], [0, 0], [0, room_size(3)], 'k-');
    plot3([room_size(1), room_size(1)], [0, 0], [0, room_size(3)], 'k-');
    plot3([room_size(1), room_size(1)], [room_size(2), room_size(2)], [0, room_size(3)], 'k-');
    plot3([0, 0], [room_size(2), room_size(2)], [0, room_size(3)], 'k-');
    
    % Plot reference point (red)
    plot3(ref_point_mic(1), ref_point_mic(2), ref_point_mic(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    % Plot microphones (green)
    for i = 1:size(mics_pos, 1)
        plot3(mics_pos(i, 1), mics_pos(i, 2), mics_pos(i, 3), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    end
    
    % Plot main speaker (blue)
    plot3(main_speaker_pos(1), main_speaker_pos(2), main_speaker_pos(3), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    
    % Draw a line between reference mic and main speaker
    plot3([ref_point_mic(1), main_speaker_pos(1)], [ref_point_mic(2), main_speaker_pos(2)], [ref_point_mic(3), main_speaker_pos(3)], 'b--');
    
    % Plot background speakers (yellow) if provided
    if ~isempty(bg_speakers_pos) % Check if bg_speakers_pos is not empty
        for i = 1:size(bg_speakers_pos, 1)
            plot3(bg_speakers_pos(i, 1), bg_speakers_pos(i, 2), bg_speakers_pos(i, 3), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'y');
            % Draw a line between reference mic and bg speaker
            plot3([ref_point_mic(1), bg_speakers_pos(i, 1)], [ref_point_mic(2), bg_speakers_pos(i, 2)], [ref_point_mic(3), bg_speakers_pos(i, 3)], 'k--');
        end
    end
    
    % Set axis labels and title
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('Room and Speaker Setup');
    
    % Create legend
    legend_items = {'Reference Point', 'Microphones', 'Main Speaker'};
    legend_handles = [plot3(NaN, NaN, NaN, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'), ...
                      plot3(NaN, NaN, NaN, 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g'), ...
                      plot3(NaN, NaN, NaN, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b')];
    
    % Add background speakers to legend if provided
    if ~isempty(bg_speakers_pos)
        legend_items{end+1} = 'Background Speaker(s)';
        legend_handles(end+1) = plot3(NaN, NaN, NaN, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'y');
    end
    
    legend(legend_handles, legend_items, 'Location', 'best');
    
    % Set view and other properties
    axis equal;
    grid on;
    view(3);
    dist_main = norm(main_speaker_pos - ref_point_mic);
    txt_main = sprintf('Dist: %.2f m', dist_main);
    text((main_speaker_pos(1) + ref_point_mic(1))/2, ...
         (main_speaker_pos(2) + ref_point_mic(2))/2, ...
         (main_speaker_pos(3) + ref_point_mic(3))/2, ...
         txt_main, 'Color', 'blue');

    % Add distance information for each background speaker
    if ~isempty(bg_speakers_pos)
        for i = 1:size(bg_speakers_pos, 1)
            dist_bg = norm(bg_speakers_pos(i,:) - ref_point_mic);
            
            % Calculate angle between main speaker and this background speaker
            vec1 = main_speaker_pos - ref_point_mic;
            vec2 = bg_speakers_pos(i,:) - ref_point_mic;
            vec1_norm = vec1 / norm(vec1);
            vec2_norm = vec2 / norm(vec2);
            dot_product = dot(vec1_norm, vec2_norm);
            dot_product = min(max(dot_product, -1), 1); % Ensure it's in valid range
            angle_deg = acos(dot_product) * (180/pi);
            
            txt_bg = sprintf('Dist: %.2f m, Angle: %.1f°', dist_bg, angle_deg);
            text((bg_speakers_pos(i,1) + ref_point_mic(1))/2, ...
                 (bg_speakers_pos(i,2) + ref_point_mic(2))/2, ...
                 (bg_speakers_pos(i,3) + ref_point_mic(3))/2, ...
                 txt_bg, 'Color', 'black');
        end
    end
    
    % 5. Speaker information (Bottom-Middle)
    subplot(2, 3, 5); 
    axis off; 
    
    main_speaker_name = speakers_filenames{1};
    
    bg_speaker_names_str = 'None'; 
    if length(speakers_filenames) > 1
        bg_speaker_names_list = speakers_filenames(2:end); 
        % Format each background speaker with a hyphen and a newline
        bg_speaker_names_str = sprintf('  %s\n', bg_speaker_names_list{:});
        bg_speaker_names_str = bg_speaker_names_str(1:end-1); % Remove the last trailing newline
    end

    full_text_str = sprintf('Speaker Information:\n\nMain Speaker: \n%s\n\nBackground Speakers:\n%s', ...
                            main_speaker_name, bg_speaker_names_str);
    
    text(0.05, 0.95, full_text_str, ...
         'Units', 'normalized', ...
         'VerticalAlignment', 'top', ...
         'HorizontalAlignment', 'left', ...
         'FontSize', 10, ...
         'Interpreter', 'none');

    sgtitle(sprintf('Room Acoustics Simulation (%.1f x %.1f x %.1f m)', room_size(1), room_size(2), room_size(3)));
    
    % 6. Plot spectrogram for first background speaker (Bottom-Middle)
    subplot(2, 3, 6);
    if ~isempty(bg_mixed_conv)
        spectrogram(bg_mixed_conv(1,:),512,128,512,'y');
        title('First BG Speaker Spectrogram');
    else
        axis off;
        text(0.5, 0.5, 'No Background Speaker', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end

    hold off; 

    
end

function is_aligned = are_speakers_aligned(ref_point, speaker1_pos, speaker2_pos, threshold_angle_deg)
% ARE_SPEAKERS_ALIGNED checks if two speakers are aligned from the perspective of a reference point
%
% Parameters:
%   ref_point - reference point (e.g., microphone) position [x y z] in meters
%   speaker1_pos - position of the first speaker [x y z] in meters
%   speaker2_pos - position of the second speaker [x y z] in meters
%   threshold_angle_deg - threshold angle in degrees below which speakers are considered aligned
%
% Returns:
%   is_aligned - boolean indicating whether the speakers are aligned (true) or not (false)

    % Default threshold if not provided (5 degrees)
    if nargin < 4
        threshold_angle_deg = 5;
    end
    
    % Convert threshold to radians
    threshold_angle_rad = threshold_angle_deg * (pi/180);
    
    % Calculate vectors from reference point to each speaker
    vec1 = speaker1_pos - ref_point;
    vec2 = speaker2_pos - ref_point;
    
    % Normalize vectors
    vec1_norm = vec1 / norm(vec1);
    vec2_norm = vec2 / norm(vec2);
    
    % Calculate dot product
    dot_product = dot(vec1_norm, vec2_norm);
    
    % Ensure dot product is within valid range for acos [-1, 1]
    dot_product = min(max(dot_product, -1), 1);
    
    % Calculate angle between vectors in radians
    angle_rad = acos(dot_product);
    
    % Check if angle is smaller than threshold
    is_aligned = (angle_rad < threshold_angle_rad);
end

function splitLists = create_data_splits(pathToFileDir, numTrain, numValidation, randomSeed)
% CREATE_DATA_SPLITS Splits a directory of .mat files into train, validation, and test sets.
%
% This function reads all .mat files from a specified directory, shuffles them,
% and then splits them into three lists based on the desired number of files
% for the training and validation sets. The remaining files form the test set.
%
% Parameters:
%   pathToFileDir    - (char array) The full path to the directory containing
%                      the .mat files. E.g., 'C:\MyData\GeneratedMats'.
%   numTrain         - (integer) The desired number of files for the training set.
%                      If this number exceeds the total available files, it will
%                      be capped at the total number of files.
%   numValidation    - (integer) The desired number of files for the validation set.
%                      If this number exceeds the remaining files after the
%                      training split, it will be capped accordingly.
%   randomSeed       - (optional, integer) An integer to initialize the random
%                      number generator for reproducible shuffling. If not provided
%                      or empty (`[]`), a new seed is generated each time.
%                      Using a fixed seed ensures the same split every run.
%
% Returns:
%   splitLists       - (struct) A structure containing three fields:
%                      .train: A list (cell array of structs from `dir`) of files
%                              assigned to the training set.
%                      .validation: A list of files assigned to the validation set.
%                      .test: A list of files assigned to the test set.
%                      Each struct in the list has fields like 'name', 'folder', etc.

allFiles = dir(fullfile(pathToFileDir, '*.mat'));
totalFiles = numel(allFiles); 

if totalFiles == 0
    warning('create_data_splits:NoMatFiles', 'No .mat files found in ''%s''. Returning empty lists.', pathToFileDir);
    splitLists.train = struct([]);
    splitLists.validation = struct([]);
    splitLists.test = struct([]);
    return;
end

if nargin < 4 || isempty(randomSeed)
    rng('shuffle');
else
    if ~isnumeric(randomSeed) || ~isscalar(randomSeed) || mod(randomSeed, 1) ~= 0
        error('create_data_splits:InvalidSeed', 'randomSeed must be an integer.');
    end
    rng(randomSeed);
end

shuffledFiles = allFiles(randperm(totalFiles));

% Cap numTrain if it exceeds total available files
numTrainActual = min(numTrain, totalFiles);
if numTrainActual < numTrain
    warning('create_data_splits:TrainCountCapped', 'Requested train count (%d) exceeds total files (%d). Train set capped to %d.', numTrain, totalFiles, numTrainActual);
end

% Cap numValidation based on remaining files after train split
remainingAfterTrain = totalFiles - numTrainActual;
numValidationActual = min(numValidation, remainingAfterTrain);
if numValidationActual < numValidation
    warning('create_data_splits:ValidationCountCapped', 'Requested validation count (%d) exceeds remaining files after train split (%d). Validation set capped to %d.', numValidation, remainingAfterTrain, numValidationActual);
end

% Test set takes all remaining files
numTestActual = totalFiles - numTrainActual - numValidationActual;

fprintf('Total files found: %d\n', totalFiles);
fprintf('Train set (requested/actual): %d/%d\n', numTrain, numTrainActual);
fprintf('Validation set (requested/actual): %d/%d\n', numValidation, numValidationActual);
fprintf('Test set (actual): %d\n', numTestActual);

% Calculate cumulative indices
idxEndTrain = numTrainActual;
idxEndValidation = idxEndTrain + numValidationActual;
idxEndTest = totalFiles; % Should be equal to totalFiles

% Assign files to lists
splitLists.train = shuffledFiles(1 : idxEndTrain);
splitLists.validation = shuffledFiles(idxEndTrain + 1 : idxEndValidation);
splitLists.test = shuffledFiles(idxEndValidation + 1 : idxEndTest);

% Handle cases where a list might be empty if requested counts sum up to totalFiles
if isempty(splitLists.train) && numTrainActual > 0
    % This handles cases where 1:0 would create an empty double array, but
    % we want an empty struct array if no files.
    splitLists.train = struct([]);
end
if isempty(splitLists.validation) && numValidationActual > 0
    splitLists.validation = struct([]);
end
if isempty(splitLists.test) && numTestActual > 0
    splitLists.test = struct([]);
end

end

function save_list_to_txt(fileList, filename)
% SAVE_LIST_TO_TXT Saves a list of file names to a text file.
%
% This function takes a struct array containing file information 
% generated by MATLAB's `dir` function and writes the 'name' field
% of each file struct to a new line in a specified text file.
%
% Parameters:
%   fileList - (struct array) A list of file information structs. Each struct
%              in the array is expected to have a 'name' field (e.g., as
%              returned by `dir`).
%   filename - (char array or string) The full path and name of the text file
%              to be created or overwritten.
%
% Returns:
%   None. This function does not return any value. It creates or overwrites
%   the specified text file as a side effect.
%
% Throws:
%   Error if the specified `filename` cannot be opened for writing (e.g.,
%   due to invalid path, insufficient permissions, or being read-only).

    fid = fopen(filename, 'w');
    if fid == -1, error('Could not open file %s for writing.', filename); end
    for i = 1:numel(fileList)
        fprintf(fid, '%s\n', fileList(i).name);
    end
    fclose(fid);
end


function saved_file_path = save_current_script_for_versioning(output_directory)
% SAVE_CURRENT_SCRIPT_FOR_VERSIONING saves the currently running .m file
% to a specified directory with a timestamp.
%
%   saved_file_path = save_current_script_for_versioning(output_directory)
%
%   Inputs:
%       output_directory (char array): The path to the directory where
%                                      the script should be saved. This
%                                      directory will be created if it
%                                      does not exist.
%
%   Outputs:
%       saved_file_path (char array): The full path to the newly saved
%                                     (versioned) script file.
%
%   Example:
%       % Call this from your main script (e.g., generate_dataset.m)
%       save_current_script_for_versioning('D:\MyProject\Versions\Scripts');
%
%   Note: This function must be called from within a saved .m file.
%         It will not work as expected if called from the MATLAB Command Window.

    % Get the full path of the currently running script
    current_script_fullpath = mfilename('fullpath');
    current_script_fullpath = sprintf('%s.m',current_script_fullpath);

    % Check if the function is called from a saved script
    if isempty(current_script_fullpath)
        error('save_current_script_for_versioning:NoScriptFound', ...
              'This function must be called from within a saved .m file to determine its path.');
    end

    % Construct the full destination path
    destination_fullpath = fullfile(output_directory);

    % Create the output directory if it doesn't exist
    if ~isfolder(output_directory)
        mkdir(output_directory);
        fprintf('Created directory: %s\n', output_directory);
    end

    % Copy the current script to the destination
    try
        copyfile(current_script_fullpath, destination_fullpath);
        fprintf('Successfully saved versioned script:\n  FROM: %s\n  TO:   %s\n', ...
                current_script_fullpath, destination_fullpath);
        saved_file_path = destination_fullpath;
    catch ME
        error('save_current_script_for_versioning:CopyFailed', ...
              'Failed to save the current script. Error: %s', ME.message);
    end
end