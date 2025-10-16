function [out] = crandn(varargin)

% Copyright 2004 The MathWorks, Inc.

try
    m1 = randn(varargin{:});
    m2 = randn(varargin{:});
    out = (m1 + sqrt(-1)*m2)/2;
catch fred
   throw(fred);
end