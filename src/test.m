function [ y ] = test( bin, data, model )

model_files = dir([model, '.*']);
[model_path, model_prefix] = fileparts(model);
command_prefix = [bin, ' --detail '];

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
    model_num = strtok(model_name(length(model_prefix)+1:end), '.');
    data_path = [data, '.', model_num, '.txt'];
    command = [command_prefix, data_path, ' ', model_path, '/', model_name];
    [status, stdout] = unix(command);
    if status ~= 0
        error(['Error during testing "', data_path, '".']);
    end
    y{i} = stdout;
end

end
