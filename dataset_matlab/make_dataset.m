clear *

% Output directory for generated files
datafolder_name = "[PATH_TO_OUTPUT_DATA_FILES]"; 

% Input directory for WSJ speech files
wsj_folder = "[PATH_TO_WSJ_DATA_FILES]";

fprintf('Starting dataset generation...\n');
fprintf('Output: %s\n', datafolder_name);
fprintf('WSJ Source: %s\n', wsj_folder);

generate_dataset(1500, datafolder_name, wsj_folder, [], [], [], [], [], []);

%% Example 1: Basic regeneration from log:

% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/test_control_from_log/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path);


%% Example 2: Regenerate from log with parameter overrides (this is a _v3a version with NFFT and FFTSHIFT modifications):
% You can override any parameter from the original generation while keeping
% % the same mix configurations (speakers, positions, SIR values, etc.)

% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/dataHSE_v3a/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'NFFT', 1024, 'FFTSHIFT', 256);

%% Example 3: Room size change -> _v3b

% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/dataHSE_v3b/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'room_size', [3 4 2.5]);

%% Example 4: Change T60 -> _v3c

% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/dataHSE_v3c/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'beta', 0.6); 

%% Example 5: Change FFT only -> _v3d
% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/dataHSE_v3d/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'NFFT', 1024);

%% Example 6: Change T60 -> _v3e
% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/dataHSE_v3e/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'beta', 0.3);

%% Example 7: Change SIR -> _v3f
% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/dataHSE_v3f/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'desired_SIR_range', [-3 3]);

%% Example 8: Change SIR -> _v3g
% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/dataHSE_v3g/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'desired_SIR_range', [-6 0]);

%% Testing -> regeneration of 7 with switched speakers. MS as a BG and BG as a MS.
% log_file_path = "D:/dataHSE_v3f/info/generation_log_regenerated.mat"; %% Original v3f
% log_file_path = "D:/generation_log_v3f_switched.mat"; %% Switched v3f
% datafolder_name_new = 'D:/dataHSE_v3f_testing_switch/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], []);
%----------
% log_file_path = "D:/dataHSE_v3/info/generation_log.mat";
% datafolder_name_new = 'D:/dataHSE_v3f_scratch/files/';
% generate_dataset([], datafolder_name_new, [], [], log_file_path, [], [], [], 'desired_SIR_range', [-3 3]);
%% Possible params that may be changed

    % % --- Parameters to log --- % % Baseline dataset is shown below.
    % params = struct();
    % params.rng_seed = rng_seed;
    % params.NFFT = 512;
    % params.FFTSHIFT = 128;
    % params.mics_num = 3;
    % params.mics = 1:params.mics_num;
    % params.record_len = 5;
    % params.max_dist = 1.25;
    % params.max_angle = pi/6;
    % params.min_dist_bg = 1.5;
    % params.num_bg_speakers = 1;
    % params.room_size = [5 6 2.5];
    % params.ref_point_mic = [2.0 1.25 1.45];
    % params.fs = 16000;
    % params.mics_pos = 0.01*[-3 0 3; 0 0 3; 3 0 3; -3 0 0; 0 0 0; 3 0 0; -3 0 -3; 0 0 -3; 3 0 -3] + params.ref_point_mic;
    % params.datafolder_name = datafolder_name;
    % params.wsj_folder = wsj_folder;
    % params.nr_mix_samples = nr_mix_samples;

    % params.numTrain = numTrain;
    % params.numValidation = numValidation;
    % params.numTest = numTest;
    % % see "function splitLists = create_data_splits(pathToFileDir, numTrain, numValidation, randomSeed)" for more info. 

    % params.desired_SNR = +inf; % Changed with removal of noise addition
    % % Note: desired_SIR will be generated per mix for variability
    % params.desired_SIR_range = [2, 10]; % 2-10 dB range for SIR

    % % Parameters for RIR_generator
    % params.c = 340;                    % Sound velocity (m/s)
    % params.fs = 16000;                 % Sample frequency (samples/s) %% Carefull here it is given only for completness. It is being set above.
    % params.L = params.room_size;       % Room dimensions [x y z] (m)
    % params.nsample = 4096;             % Number of samples
    % params.beta = 0.18;                % Reverberation time (s)
    % params.mtype = 'hypercardioid';    % Type of microphone
    % params.order = -1;                 % -1 equals maximum reflection order!
    % params.dim = 3;                    % Room dimension
    % params.orientation = [pi/2 0];     % Microphone orientation (rad)
    % params.hp_filter = 1;              % Enable high-pass filter (1=enable, 0=disable)