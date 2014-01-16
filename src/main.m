train_data_name_prefix = 'caltech101';
test_data_name_prefix = 'caltech101';
D_subspace = 100;
N_bootstrap = 5;

config_path = fullfile('..', 'config');


train_bin_path = fullfile('..', 'bin', 'mytree_train');
test_bin_path = fullfile('..', 'bin', 'mytree_test');
train_data_path_prefix = fullfile('..', 'data', train_data_name_prefix);
test_data_path_prefix = fullfile('..', 'data', test_data_name_prefix);

% training
disp('Converting training data ... ');
param = data_convert(train_data_path_prefix, X, Y, D_subspace, N_bootstrap);
train(train_bin_path, train_data_path_prefix, config_path);

% testing
disp('Converting testing data ... ');
data_convert(test_data_path_prefix, X_, [], param);
disp('Testing data ... ');
y = test(test_bin_path, test_data_path_prefix, train_data_path_prefix);
Y_stat = test_evaluate(y, param.n_label, size(Y_, 2));

[Y_stat_max, Y_test] = max(Y_stat);
n_correct = length(find(Y_test == Y_));
Y_stat_ismax = bsxfun(@eq, Y_stat_max, Y_stat);
Y_not_unique_max = find(sum(Y_stat_ismax) > 1);
n_correct_worst = n_correct;
n_correct_best = n_correct;
for i = Y_not_unique_max
    label_of_max = find(Y_stat_ismax(:, i));
    if Y_test(i) == Y_(i)
        n_correct_worst = n_correct_worst - 1;
    elseif ~isempty(find(label_of_max == Y_(i), 1))
        n_correct_best = n_correct_best + 1;
    end
end
acc_best = n_correct_best / size(Y_, 2);
acc_worst = n_correct_worst / size(Y_, 2);

disp('Done.');