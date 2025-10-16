root = 'D:\wsj0';

unique_map = containers.Map();

all_files = dir([root '\**']);

load("used_wavs.mat"); % wavs that were used.

files_only = arrayfun(@(x) ~x.isdir,all_files);
files_only_indx = find(files_only);

non_dir_files = all_files(files_only_indx);

old_wavs_only = non_dir_files(endsWith({non_dir_files.name}, '.wv1'));

for i = 1:length(old_wavs_only)
    current_wav = old_wavs_only(i);
    full_path = fullfile(current_wav.folder, current_wav.name);
    unique_map(current_wav.name) = full_path;
end

%%
is_file = ~[used_files.isdir];
used_names = {used_files(is_file).name};
used_basenames = regexprep(used_names, '\.[^.]+$','');

unique_keys = keys(unique_map);
k_basenames = regexprep(unique_keys, '\.[^.]+$','');

tf = ismember(k_basenames, used_basenames);
this_was_used = unique_keys(tf);

used_paths = values(unique_map, this_was_used);
% used_paths_no_root = cellfun(@(x) x(8:end), used_paths, 'UniformOutput', false); % removes the root from the path of .wv1 files
%%

output_folder = fullfile('D:\unique_wsj0');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Copy our 3557 used files from original wsj0 file structure into a selected output file.
for i = 1:numel(used_paths)
    recording_path = used_paths{i};
    recording_name = strsplit(recording_path,filesep, 'CollapseDelimiters', true);
    copyfile(recording_path, fullfile(output_folder,recording_name{end}));
end

