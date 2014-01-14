function [ param ] = data_convert( prefix, X, Y, param )
% Convert data matrix to txt files in libsvm/svmlight format.
% Usage:
%   [ param ] = data_convert(prefix, X, Y, D_subspace)
%       prefix    : The file name's prefix.
%       X         : An D-by-N data matrix. (N: num of samples, D: dimension)
%       Y         : An 1-by-N label matrix.
%       D_subspace: Dimension of subspaces.
%       param     : Parameters of conversion.
%   [] = data_convert(prefix, X, Y, param)
%       prefix    : The file name's prefix.
%       X         : An D-by-N data matrix. (N: num of samples, D: dimension)
%       Y         : An 1-by-N label matrix.
%       param     : Parameters of conversion.

narginchk(3, 4);

D = size(X, 1);
if nargin == 3 || ~isstruct(param)
    if size(X, 2) ~= length(Y)
        error('dimension mismatch.');
    end
    if nargin == 3
        D_subspace = 1;
    else
        D_subspace = param;
    end
    [X, param] = data_scale(X);
    param.D = D;
    param.n_label = max(Y);
    param.D_subspace = D_subspace;
    param.subspace = randperm(D);
else
    X = data_scale(X, param);
    D_subspace = param.D_subspace;
end
n_subspace = floor(D / D_subspace);

X = X(param.subspace, :);
X_subspace = cell(n_subspace, 1);
for i = 1 : n_subspace
    X_subspace{i} = X((i-1)*D_subspace+1 : i*D_subspace, :);
end

n_digits = num2str(floor(log10(n_subspace)) + 1);
parfor i = 1 : n_subspace
    filename = sprintf(['%s_%0',n_digits,'d.txt'], prefix, i);
    fid = fopen(filename, 'wt');
    if fid == -1
        error('error opening the file:');
    end
    convert(fid, X_subspace{i}, Y);
    fclose(fid);
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
