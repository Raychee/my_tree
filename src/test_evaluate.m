function [ Y ] = test_evaluate( y, n_label, n_sample )

Y = zeros(n_label, n_sample);
for i = 1 : length(y)
    test_sample = strsplit(y{i}, '\n');
    i_sample = 0;
    for j = 1 : length(test_sample)
        if isempty(test_sample{j})
            continue;
        end
        i_sample = i_sample + 1;
        if i_sample > n_sample
            error('Number of samples mismatch');
        end
        [~, sample_distrib] = strtok(test_sample{j});
        sample_distrib = sscanf(sample_distrib, '%d:%d', [2, inf]);
        Y(sample_distrib(1, :), i_sample) = ...
                Y(sample_distrib(1, :), i_sample) + sample_distrib(2, :)';
    end
    if i_sample ~= n_sample
        error('Number of samples mismatch');
    end
end

end
