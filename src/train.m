function [] = train( bin, config, data, model, n_subspace, n_bootstrap, n_parallel )

command = ['parallel --progress --delay 1 -j ', num2str(n_parallel)];

if n_bootstrap == 0
    command = [command, ' "', bin, ' --config ', config, ...
               ' -o ', model, '.{} ', ...
               data, '.{}.txt >', data, ...
               '.{}.log 2>&1" ::: `seq -w 1 ', ...
               num2str(n_subspace), '`'];
else
    s_sub_bootstrap = ceil(n_bootstrap / ceil(n_parallel / n_subspace));
    n_sub_bootstrap = floor(n_bootstrap / s_sub_bootstrap);
    s_rem_bootstrap = n_bootstrap - s_sub_bootstrap * n_sub_bootstrap;
    commandfile = tempname;
    fid = fopen(commandfile, 'wt');
    if fid == -1
        error(['error creating temporary file: ', commandfile]);
    end
    n_digits_subspace = num2str(floor(log10(n_subspace)) + 1);
    n_digits_sub_bootstrap = num2str(floor(log10(n_sub_bootstrap)) + 1);
    for i = 1 : n_sub_bootstrap
        for j = 1 : n_subspace
            i_bootstrap = (i - 1) * s_sub_bootstrap + 1;
            data_name = [data, '.', sprintf(['%0', n_digits_subspace, 'd'], j)];
            model_name = [model, '.', sprintf(['%0', n_digits_subspace, 'd'], j)];
            log_name = [data_name, '.', sprintf(['%0', n_digits_sub_bootstrap, 'd'], i)];
            bin_command = [bin, ' --config ', config, ' -o ', model_name, ...
                           ' --bootstrap ', num2str(s_sub_bootstrap), ...
                           ' --bootstrap-number-starts-by ', num2str(i_bootstrap), ...
                           ' ', data_name, '.txt >', ...
                           log_name, '.log 2>&1'];
            fprintf(fid, [bin_command, '\n']); 
        end
    end
    if s_rem_bootstrap > 0
        for j = 1 : n_subspace
            i_bootstrap = s_sub_bootstrap * n_sub_bootstrap + 1;
            data_name = [data, '.', sprintf(['%0', n_digits_subspace, 'd'], j)];
            model_name = [model, '.', sprintf(['%0', n_digits_subspace, 'd'], j)];
            log_name = [data_name, '.', sprintf(['%0', n_digits_sub_bootstrap, 'd'], n_sub_bootstrap + 1)];
            bin_command = [bin, ' --config ', config, ' -o ', model_name, ...
                           ' --bootstrap ', num2str(s_rem_bootstrap), ...
                           ' --bootstrap-number-starts-by ', num2str(i_bootstrap), ...
                           ' ', data_name, '.txt >', ...
                           log_name, '.log 2>&1'];
            fprintf(fid, [bin_command, '\n']); 
        end
    end
    fclose(fid);
    command = [command, ' <', commandfile];
end

if unix(command, '-echo') ~= 0
    error('Error during training.');
end
       
end

