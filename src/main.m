bin_train   = '../bin/mytree_train';
bin_test    = '../bin/mytree_test';
data_dir    = '../data/';
config_file = '../config';

train_file_name = 'caltech_train';
test_file_name  = 'caltech_test';
model_dir_name  = 'caltech_model';

d_subspace = 100;
n_bootstrap = 5;

n_parallel = 20;

n_label = max(Y);
n_sample = length(Y);

disp('Converting training data.');
[X, scale, offset] = data_scale(X);
if d_subspace > 0
    n_subspace = floor(size(X, 1) / d_subspace);
    subspace = randperm(size(X, 1));
    X = X(subspace, :);
else
    d_subspace = size(X, 1);
    n_subspace = 1;
end
data_convert([data_dir, train_file_name], X, Y, d_subspace, n_subspace);

disp('Converting testing data.');
X_ = data_scale(X_, scale, offset);
if d_subspace < size(X, 1)
    X_ = X_(subspace, :);
end
data_convert([data_dir, test_file_name], X_, Y_, d_subspace, n_subspace);

clear('X');
clear('X_');

disp('Training.');
train(bin_train, config_file, [data_dir, train_file_name], ...
      [data_dir, model_dir_name], n_subspace, n_bootstrap, n_parallel);

disp('Testing.');
y_train = test(bin_test, [data_dir, train_file_name], [data_dir, model_dir_name]);
y_test = test(bin_test, [data_dir, test_file_name], [data_dir, model_dir_name]);
Y_train = test_evaluate(y_train, n_label, n_sample);
Y_test = test_evaluate(y_test, n_label, n_sample);

[acc_best_train, acc_worst_train, n_acc_best_train, n_acc_worst_train] ...
    = test_stat(Y_train, Y);
[acc_best_test, acc_worst_test, n_acc_best_test, n_acc_worst_test] ...
    = test_stat(Y_test, Y_);

disp('Done.');
disp(['Accuracy on testing data: ', ...
     num2str(acc_best_test*100), '%(best), ', ...
     num2str(acc_worst_test*100), '%(worst)']);
disp(['Accuracy on training data: ', ...
     num2str(acc_best_train*100), '%(best), ', ...
     num2str(acc_worst_train*100), '%(worst)']);