function [w, a, soi, NumIt] = iFICA(x, wini, soiinfo, nonln)

epsilon = 0.000001;
[d, N, K] = size(x);
MaxIt = 100;
realvalued = isreal(x);

if nargin < 4
   nonln = 'rati';
end

if size(soiinfo,1)>1
    soiinfo = permute(soiinfo,[3 2 1]);
end

Cx = pagemtimes(x,'none',x,'ctranspose')/N;

%one-unit FastICA
%w = Cx\aini;
%w = pagemldivide(Cx,aini);
w = wini;
w = w./sqrt(sum(w.*conj(w),1));
NumIt = 0;
crit = 0;

if nargin > 2
    H = pagemtimes((x./(abs(soiinfo)+0.001)), 'none', x, 'ctranspose')/N;
else
    H = Cx;
end

while crit < 1-epsilon && NumIt < MaxIt
    NumIt = NumIt + 1;
    wold = w; 
    a = pagemtimes(Cx, 'none', w, 'none'); 
    sigma2 = sum(conj(w).*a, 1); % variance of SOI on blocks
    a = a./sigma2; % mixing vector
    soi = pagemtimes(w, 'ctranspose', x, 'none');
    sigma = sqrt(sigma2); 
    soin = soi./sigma; % normalized SOI 
    if realvalued
        [psi, psihpsi] = realnonln(soin, nonln);
    else
        [psi, psihpsi] = complexnonln(soin, nonln);
    end
    xpsi = (pagemtimes(x, 'none', psi, 'transpose')./sigma)/N;
    rho = mean(psihpsi,2);
    grad_a = rho.*a - xpsi;
    w = pagemldivide(H, grad_a);
    w = w./sqrt(sum(w.*conj(w),1));
    crit = min(abs(sum(w.*conj(wold),1)),[],3);
end
 
a = pagemtimes(Cx, 'none', w, 'none'); 
sigma2 = sum(conj(w).*a, 1);
a = a./sigma2; 
soi = a(1,1,:).*pagemtimes(w, 'ctranspose', x, 'none');
w = conj(a(1,1,:)).*w;
a = a./a(1,1,:);

end


%%%%%%%%%%% helping functions

function [psi, psipsi] = realnonln(s,nonln)
    if strcmp(nonln,'sign')
        if size(s,3)==1, error('Nonlinearity "sign" cannot be used for the real-valued ICA/ICE.'); end
        aux = 1./sqrt(sum(s.^2,3));
        psi = s.*aux;
        psipsi = aux.*(1-psi.^2);
    elseif strcmp(nonln,'tanh')
        if size(s,1) > 1
            aux = 1./sqrt(sum(s.^2,3));
            th = tanh(s);
            psi = th.*aux;
            psipsi = aux.*(1 - th.^2 - psi.*aux);
        else
            psi = tanh(s);
            psipsi = 1 - psi.^2;
        end
    elseif strcmp(nonln,'rati')
        aux = 1./(1+sum(s.^2,3));
        psi = s.*aux;
        psipsi = aux - 2*psi.^2;
    end
end

function [psi, psipsi] = complexnonln(s,nonln)
    if strcmp(nonln,'sign')
        sp2 = s.*conj(s);
        aux = 1./sqrt(sum(sp2,3));
        psi = conj(s).*aux;
        psipsi = aux.*(1-psi.*conj(psi)/2);
    elseif strcmp(nonln,'rati')
        sp2 = s.*conj(s);
        aux = 1./(1+sum(sp2,3));
        psi = conj(s).*aux;
        psipsi = aux - psi.*conj(psi);        
    end
end