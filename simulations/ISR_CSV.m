function out = ISR_CSV(w,x,n)

if size(w,2)>1
    w = permute(w,[1 3 2]);
end

signal = pagemtimes(w,'ctranspose',x-n,'none');
interference = pagemtimes(w,'ctranspose',n,'none');

out = squeeze(sum(interference.*conj(interference),2)./sum(signal.*conj(signal),2));
