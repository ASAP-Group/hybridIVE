function [w, a, soi, NumIt] = ifastive(x, aini, soiinfo, nonln, iotta)

epsilon = 0.000001;
[d, N, K] = size(x);
MaxIt = 100;
realvalued = isreal(x);

if nargin < 5
    iotta = 0.001;
end

if nargin < 4
   nonln = 'rati';
end

if nargin>2 
    if size(soiinfo,1)>1
        soiinfo = permute(soiinfo,[3 2 1]);
    end
end

Cx = pagemtimes(x,'none',x,'ctranspose')/N;


a = aini;

NumIt = 0;
crit = 0;

if nargin > 2
    alphaweights = 1./(iotta+abs(soiinfo));
else
    alphaweights = ones(1,N);
end
alphaweights = alphaweights./mean(alphaweights,2);
Cw = pagemtimes(x.*alphaweights, 'none', x, 'ctranspose')/N;
Cwinv = pageinv(Cw);

w = pagemtimes(Cwinv,'none',a,'none');
sigmaw2 = real(1./pagemtimes(a,'ctranspose',w,'none'));
w = w.*sigmaw2;

a = pagemtimes(Cx,'none',w,'none');
a = a./pagemtimes(a,'ctranspose',w,'none');

while crit < 1-epsilon && NumIt < MaxIt
    NumIt = NumIt + 1;
    aold = a; 
    
    % SOI normalization
    sigmax2 = real(sum(conj(w).*pagemtimes(Cx,'none',w,'none'),1)); % w'*Cx*w; % variance of SOI on blocks
    soi = pagemtimes(w, 'ctranspose', x, 'none');
    sigma = sqrt(sigmax2); 
    soin = soi./sigma; % normalized SOI 

    % nonlinearity
    if realvalued
        [psi, psihpsi] = realnonln(soin, nonln);
    else
        [psi, psihpsi] = complexnonln(soin, nonln);
    end

    % update
    xpsi = (pagemtimes(x, 'none', psi, 'transpose')./sigma)/N;
    rho = mean(psihpsi,2);
    nu = pagemtimes(w,'ctranspose',xpsi,'none'); % w'*xpsi;


    grad = a - xpsi./nu;

    a = a - grad./(1-(rho./nu)); 
    
    w = pagemtimes(Cwinv,'none',a,'none');
    sigmaw2 = real(1./pagemtimes(a,'ctranspose',w,'none'));
    w = w.*sigmaw2;
    
    a = pagemtimes(Cx,'none',w,'none');
    a = a./pagemtimes(a,'ctranspose',w,'none');

    crit = min(abs(sum(a.*conj(aold),1))./sqrt(sum(a.*conj(a),1).*sum(aold.*conj(aold),1)),[],3);
end
 
% rescaling
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