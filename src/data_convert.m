function [ param ] = data_convert( prefix, X, Y, param, n_bootstrap )
% Convert data matrix to txt files in libsvm/svmlight format.
% Usage:
%   [ param ] = data_convert(prefix, X, Y, D_subspace, n_bootstrap)
%       prefix     : The file name's prefix.
%       X          : An D-by-N data matrix. (N: num of samples, D: dimension)
%       Y          : An 1-by-N label matrix.
%       D_subspace : Dimension of subspaces.
%       n_bootstrap: Number of bootstraps.
%       param      : Parameters of conversion.
%   [] = data_convert(prefix, X, Y, param)
%       prefix     : The file name's prefix.
%       X          : An D-by-N data matrix. (N: num of samples, D: dimension)
%       Y          : An 1-by-N label matrix.
%       param      : Parameters of conversion.

narginchk(3, 5);

[D, N] = size(X);
if nargin == 4 && isstruct(param)
    X = data_scale(X, param);
    D_subspace = param.D_subspace;
    n_bootstrap = 0;
else
    if size(X, 2) ~= length(Y)
        error('dimension mismatch.');
    end
    if nargin < 5
        n_bootstrap = 0;
    end
    if nargin < 4
        D_subspace = 1;
    else
        D_subspace = param;
    end
    [X, param] = data_scale(X);
    param.D = D;
    param.n_label = max(Y);
    param.D_subspace = D_subspace;
    param.n_bootstrap = n_bootstrap;
    param.subspace = randperm(D);
end
n_subspace = floor(D / D_subspace);

X = X(param.subspace, :);

n_digits_subspace = num2str(floor(log10(n_subspace)) + 1);
if n_bootstrap == 0
    n_digits_bootstrap = 0;
else
    n_digits_bootstrap = num2str(floor(log10(n_bootstrap)) + 1);
end

X_subspace = cell(n_subspace, 1);
for i = 1 : n_subspace
    X_subspace{i} = X((i-1)*D_subspace+1 : i*D_subspace, :);
end
parfor i_subspace = 1 : n_subspace
    if n_bootstrap == 0
        filename = sprintf(['%s_%0', n_digits_subspace, 'd.txt'], prefix, i_subspace);
        fid = fopen(filename, 'wt');
        if fid == -1
            error(['error opening the file: ', filename_prefix, '_train.txt']);
        end
        convert(fid, X_subspace{i_subspace}, Y);
        fclose(fid);
    else
        for i_bootstrap = 1 : n_bootstrap
            index_bootstrap = randi(N, 1, N);
            index_remain = setdiff(1:N, index_bootstrap);
            X_subsample = X_subspace{i_subspace}(:, index_bootstrap);
            Y_subsample = Y(index_bootstrap);
            X_subremain = X_subspace{i_subspace}(:, index_remain);
            Y_subremain = Y(index_remain);
            filename_prefix = sprintf(['%s_%0', n_digits_subspace, 'd_%0',n_digits_bootstrap,'d'], prefix, i_subspace, i_bootstrap);
            fid = fopen([filename_prefix, '_train.txt'], 'wt');
            if fid == -1
                error(['error opening the file: ', filename_prefix, '_train.txt']);
            end
            convert(fid, X_subsample, Y_subsample);
            fclose(fid);
            fid = fopen([filename_prefix, '_val.txt'], 'wt');
            if fid == -1
                error(['error opening the file: ', filename_prefix, '_val.txt']);
            end
            convert(fid, X_subremain, Y_subremain);
            fclose(fid);
        end
    end
end

end

function [] = convert( fid, X, Y )

for i = 1 : size(X, 2)
    if isempty(Y)
        fprintf(fid, '0');
    else
        fprintf(fid, '%d', Y(i));
    end
    for j = find(X(:, i))'
        fprintf(fid, ' %d:%.16g', j, X(j, i));
    end
    fprintf(fid, '\n');
end

end

function [ X, param ] = data_scale( X, param )

if nargin < 2
    minX = min(X, [], 2);
    maxX = max(X, [], 2);
    param.scale = 1 ./ (maxX - minX);
    param.offset = -minX;
end
X = bsxfun(@times, bsxfun(@plus, X, param.offset), param.scale);

end
