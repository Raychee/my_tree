function [ y ] = test( bin_command, data_prefix, model_path_prefix )

model_files = dir([model_path_prefix, '_*']);
[model_path, model_prefix] = fileparts(model_path_prefix);
command_prefix = [bin_command, ' --detail '];

model_names = cell(length(model_files), 1);
n_model = 0;
for i = 1 : length(model_files)
    if model_files(i).isdir
        n_model = n_model + 1;
        model_names{n_model} = model_files(i).name;
    end;
end
y = cell(n_model, 1);

parfor i = 1 : n_model
    model_name = model_names{i};
    model_num = sscanf(model_name, ...
                       ['%*',num2str(length(model_prefix)),'c_%s'], 1);
    data_path = [data_prefix, '_', model_num, '.txt'];
    command = [command_prefix, data_path, ' ', ...
               fullfile(model_path, model_name)];
    [status, stdout] = unix(command);
    if status ~= 0
        error(['Error during testing "', model_name, '".']);
    end
    y{i} = stdout;
end

end
