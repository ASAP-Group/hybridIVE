function [w, a, soi, NumIt] = iFICAGauss(x, L, wini, soiinfo)

epsilon = 0.000001;
[d, N, K] = size(x);
MaxIt = 100;

Ns = floor(N/L);
N = Ns*L;

X = permute(reshape(x(:,1:N,:), [d Ns L K]), [1 2 4 3]);

if nargin>3
    if length(soiinfo)<N
        soiinfo = kron(soiinfo(:),ones(N/length(soiinfo),1));
    else
        soiinfo = soiinfo';
        soiinfo = reshape(abs(soiinfo),Ns,L,K);
    end
    Clweighted = pagemtimes(X./permute(soiinfo+0.001,[4 1 3 2]),'none',X,'ctranspose')/Ns;
end


Cl = pagemtimes(X,'none',X,'ctranspose')/Ns;
C = mean(Cl,4);

%w = Cx\aini;
%w = pagemldivide(C,aini);
w = wini;
w = w./sqrt(sum(w.*conj(w),1));
NumIt = 0;
crit = 0;

while crit < 1-epsilon && NumIt < MaxIt
    NumIt = NumIt + 1;
    wold = w; 
    al = pagemtimes(Cl, 'none', w, 'none');
    a = mean(al,4);
    sigma2 = sum(conj(w).*al, 1); % variance of SOI 
    a = a./mean(sigma2,4); % mixing vector
    if nargin > 3
        H = mean(Clweighted./sigma2,4);
    else
        H = mean(Cl./sigma2,4);
    end
    w = pagemldivide(H, a);
    w = w./sqrt(sum(w.*conj(w),1));
    crit = min(abs(sum(w.*conj(wold),1)),[],3);
end
 
a = pagemtimes(C, 'none', w, 'none'); 
sigma2 = sum(conj(w).*a, 1);
a = a./sigma2; 
soi = a(1,1,:).*pagemtimes(w, 'ctranspose', x, 'none');
w = conj(a(1,1,:)).*w;
a = a./a(1,1,:);

end


