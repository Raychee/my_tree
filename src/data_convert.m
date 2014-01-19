function [] = data_convert( prefix, X, Y, d_subspace, n_subspace )
% [] = data_convert( prefix, X, Y, d_subspace, n_subspace )

narginchk(3, 5);

if nargin == 3
    d_subspace = size(X, 1);
    n_subspace = 1;
elseif nargin == 4
    n_subspace = floor(size(X, 1) / d_subspace);
end

X_subspace = cell(n_subspace, 1);
for i = 1 : n_subspace
    X_subspace{i} = X((i-1)*d_subspace+1 : i*d_subspace, :);
end

n_digits = num2str(floor(log10(n_subspace)) + 1);
parfor i = 1 : n_subspace
    filename = [prefix, sprintf(['.%0', n_digits, 'd.txt'], i)];
    fid = fopen(filename, 'wt');
    if fid == -1
        error(['error opening the file: ', filename]);
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

