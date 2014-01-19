function [ varargout ] = data_scale( X, varargin )
% [X, scale, offset] = data_scale(X)
% [X] = data_scale(X, scale, offset)

if nargin == 1
    minX = min(X, [], 2);
    maxX = max(X, [], 2);
    scale = 1 ./ (maxX - minX);
    offset = -minX;
    varargout = cell(3, 1);
    varargout{2} = scale;
    varargout{3} = offset;
elseif nargin == 3
    scale = varargin{1};
    offset = varargin{2};
else
    error('Number of inputs not allowed');
end

varargout{1} = bsxfun(@times, bsxfun(@plus, X, offset), scale);

end