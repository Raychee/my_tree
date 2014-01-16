function [] = train(bin_command, data_prefix, config_path)

narginchk(2, 3);

train_files = dir([data_prefix, '_*_*_train.txt']);
path = fileparts(data_prefix);
if nargin == 2
    command_prefix = bin_command;
else
    command_prefix = [bin_command, ' --config ', config_path];
end

parfor i = 1 : length(train_files)
    if train_files(i).isdir
        continue;
    end
    file_name = train_files(i).name;
    suffix = strfind(file_name, '_train.txt');
    model_name = [file_name(1:suffix(end)), 'model'];
    model_path = fullfile(path, model_name);
    file_path = fullfile(path, file_name);
    if exist(model_path, 'dir')
        delete(fullfile(model_path,'*.node'));
    else
        if ~mkdir(model_path)
            error(['Failed creating folder "', model_path, '".']);
        end
    end
    command = [command_prefix, ' -o ', model_path, filesep, ...
               ' ', file_path, ' >', model_path, filesep, 'log.txt 2>&1'];
    disp(['Begin: Training model ', model_name, '.']);
    if unix(command) ~= 0
        error(['Error during training "', model_name, '".']);
    end
    disp(['End: Training model ', model_name, '.']);
end

end