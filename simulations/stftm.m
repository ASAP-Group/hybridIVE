function X = stftm(x, n, shift, nfft, win)

[N, d] = size(x);
if d > N
    x = x';
    d = N;
end
    
for i = 1:d
    X(:,:,i) = stft(x(:,i)', n, shift, nfft, win);
end

X = permute(X, [3 2 1]);