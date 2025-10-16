function x = claplace(m,n,o)

if nargin < 2
    n = m;
end

if nargin < 3
    o = 1;
end

x = log(rand(m,n,o)).*sign(rand(m,n,o)-0.5);
x = x .* exp(1i*rand(size(x))*2*pi);