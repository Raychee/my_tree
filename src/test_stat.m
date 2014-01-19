function [ acc_best, acc_worst, n_correct_best, n_correct_worst ] = ...
         test_stat( Y_stat, Y )

[Y_stat_max, Y_test] = max(Y_stat);
n_correct = length(find(Y_test == Y));
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
acc_best = n_correct_best / size(Y, 2);
acc_worst = n_correct_worst / size(Y, 2);

end

