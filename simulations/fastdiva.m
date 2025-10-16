function [Wt, At, S, NumIt, ISRest1u] = fastdiva(x, params)
% FastDIVA: [Wt, At, S, NumIt, ISRest1u] = FastDIVA(X, PARAMS)
% 
% Extended FastICA algorithm for dynamic independent vector/component
% extraction/analysis. Piecewise linear and determined mixing model with 
% constant separating vectors (CSV) over blocks of data is enabled.
%
% This implementation requires the MTIMESX package [3]
%
% version: 2.0  release: February 7, 2022
%
% Author(s): Zbynek Koldovsky
% Technical University of Liberec
% Studentská 1402/2, LIBEREC
% Czech Republic
%
% References
%
% [1] Z. Koldovsky, V. Kautsky, P. Tichavsky, J. Cmejla, and J. Malek, 
% "Dynamic Independent Component/Vector Analysis: Time-Variant Linear 
% Mixtures Separable by Time-Invariant Beamformers", IEEE Trans. on Signal 
% Processing,  vol. 69, pp. 2158-2173, March 2021. 
%
% [2] Z. Koldovsky, V. Kautsky, and P. Tichavsky, "Double Nonstationarity: 
% Blind Extraction of Independent Nonstationary Vector/Component from 
% Nonstationary Mixtures - Algorithms", in preparation, February
% 2022.
%
% [3] J. Tursa (2020). MTIMESX - Fast Matrix Multiply with Multi-Dimensional 
% Support (https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support), 
% MATLAB Central File Exchange. 
%
% Inputs:
%
% X     observed mixture of signals d x N x K where d is the number of 
%       sensors, N is the number of samples, and K is the number of mixtures 
%       (datasets or frequencies in audio). K=1 corresponds to independent
%       component extraction/analysis (ICE/ICA) while K>1 means independent
%       vector extraction/analysis (IVE/ICE)
%
% PARAMS     struct of parameters 
%
% Outputs:
%
% Wt    array d x r x K x T contains the estimated separating 
%       vectors/matrices; r is the number of signals to be extracted; T is
%       the number of blocks
%
%       MIND THE TRANSPOSITION WHEN USING Wt! For example, the extracted
%       signals on the t-th block in the k-th data set are obtained through
%       St(:,:,k,t) = Wt(:,:,k,t)' * X(:,:,k,t)
%
% At    array  d x r x K x T contains the estimated mixing
%       vectors/matrices; 
%
% S     contains the extracted independent signals in the form that is
%       determined by 'scaling'; see below
% 
% NumIt     1 x r array containing the number of iterations per component
%
% ISRest1u  r x K array whose (i,k)-th element contains theoretical mean
%           Interference-to-Signal Ratio (ISR) of the i-th output signal in 
%           from the k-th dataset if this is extracted by the one-unit 
%           approach. This value is based the asymptotical performance
%           analysis in [1] (Equation 89) and might serve as information 
%           about the achieved extraction accuracy.
%
% Fields in PARAMS:
% -----------------
%
%   'approach' - APPROACH should contain one character corresponding to 
%               the variant of FastDIVA. The default value is 'd'.
%       
%       'u' - one-unit approach performing independent extractions of 'numsig'
%       signals (from each data set) based on the CSV model; most typical
%       is to extract 'numsig' = 1 signal by means of this approach (Blind 
%       Source Extraction)
%
%       's' - symmetric approach for dynamic ICA/IVA performing parallel
%       extractions of 'numsig' signals from each of K mixtures under the 
%       orthogonality constraint (orthogonality of entire data)
%
%       'd' - block-deflation approach performing subsequent 
%       extractions of 'numsig' signals from each of K mixtures under the 
%       orthogonality constraint (orthogonality within each block)
%
%   'hessian' - either 'fastdiva' (default) or 'quickive'; the former
%   choice is faster, however, might be less stable with Gaussian sources.
%
%   'numsig' - determines r, which is the number of signals to be extracted 
%   from each of K mixtures. If not specified by user or through INI, the 
%   default value is r = d.
%
%   'ini' - d x r x K x T vector/matrix/tensor contains the initial values 
%   of separating/mixing vectors/matrices. The content of INI is
%   interpreted depending on 'initype' and on its dimensions, where d is
%   the number of sensors (input signals), r is the number of signals to be
%   extracted (if 'numsig' is specified, its value wins, otherwise set to 
%   r), K is the number of mixtures, and T is the number of blocks. The
%   default value of INI is random.
%
%   'initype' - contains one character 'a' or 'w' determining the type of
%   initialization:
%
%       'a' - initialization by mixing vectors/matrices
%       
%       'w' - initialization by separating vectors/matrices (default)
%   
% 	'T' - an integer value determining the number of blocks in the CSV 
%   mixing model; the default value is 1, which corresponds to the 
%   conventional static mixing model. If a value T > 1 is specified, blocks 
%   of the same length Nb = floor(N/T) are considered. 
%
% 	'L' - an integer value determining the number of sub-blocks in the 
%   nonstationary-and-nonGaussianity source model; every block is divided
%   into L sub-blocks; the default value is L = 1; only sub-blocks 
%   of the same length Ns = Nb/L are implemented. 
%
%   'nonln' - model score function (nonlinearity). The default nonlinearity 
%   is 'rati'. The user is encouraged to implement its own nonlinearity
%   tailored to the data to be analyzed.
%
%   'precond' - PRECOND determines the preprocessing (preconditioning)
%   d x d transform matrix applied to the input data (can be different for 
%   each mixture). If not defined (default setting), the prewhittening 
%   transform is computed and applied to each of K mixtures. If contains
%   the empty value, i.e. [], no preconditioning is applied (the identity 
%   matrix).
%
%   'maxit' - the maximum number of iterations; the default value is 100
%
%   'scaling' - determines the scales of output signals, which is ambigual
%   in blind source separation; the value is a character
%
%       'n' - normalized scales (unit sample-based variance over all 
%       samples); this is default; S is array r x N x K 
%
%       'i' - images of extracted signals on sensors (inputs); S is array
%       d x N x K x r; At and Wt are scaled as for the option 'n'
%
%       '1' - images of extracted signals on the first sensor; S is array
%       r x N x K; At and Wt are rescaled accordingly
%
% This is unpublished proprietary source code of TECHNICAL UNIVERSITY OF
% LIBEREC, CZECH REPUBLIC.
% 
% The purpose of this software is the dissemination of scientific work for
% scientific use. The commercial distribution or use of this source code is
% prohibited. The copyright notice does not evidence any actual or intended
% publication of this code. Term and termination:
% 
% This license shall continue for as long as you use the software. However,
% it will terminate if you fail to comply with any of its terms and
% conditions. You agree, upon termination, to discontinue using, and to
% destroy all copies of, the software.  Redistribution and use in source and
% binary forms, with or without modification, are permitted provided that
% the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer (Limitation of
% warranties and liability). Redistributions in binary form must reproduce
% the above copyright notice, this list of conditions and the following
% disclaimer (Limitation of warranties and liability) in the documentation
% and/or other materials provided with the distribution. Neither name of
% copyright holders nor the names of its contributors may be used to endorse
% or promote products derived from this software without specific prior
% written permission.
% 
% The limitations of warranties and liability set out below shall continue
% in force even after any termination.
% 
% Limitation of warranties and liability:
% 
% THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
% WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE HEREBY
% DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
% INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS  OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
% LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
% OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
% SUCH DAMAGE.

if exist('pagemtimes','builtin')
    mtimesx = @newmtimesx;
end

epsilon = 1e-5; %0.00001; % stopping threshold

[d, N, K] = size(x);

realvalued = isreal(x);

if nargin < 2
    params = struct;
end

if ~isfield(params,'hessian')
    fasthessian = true;
else
    fasthessian = strcmp(params.hessian,'fastdiva');
end

if ~isfield(params,'maxit')
    params.maxit = 100;
end


if ~isfield(params,'scaling')
    params.scaling = 'n'; % normalized components 
end

if ~isfield(params,'initype')
    params.initype = 'w';
elseif ~(params.initype=='w' || params.initype=='a')
    error('FastDIVA: Invalid type of initialization.')
end

if isfield(params,'numsig')
    r = params.numsig;
elseif isfield(params,'ini')
    r = size(params.ini,2);
else
    r = d;
end

if ~isfield(params,'approach')
    if r == 1
        params.approach = 'u';
    else
        params.approach = 'd';
    end
end


if ~isfield(params,'nonln')
    params.nonln = 'rati';
end

gauss = strcmp(params.nonln,'gauss') || strcmp(params.nonln,'gausstri');

if ~isfield(params,'T')
    T = 1; % static mixing model
else
    T = params.T;
end

if ~isfield(params,'L')
    L = 1; % static mixing model
else
    L = params.L;
end

if L>1 && params.approach ~= 'u'
    warning(['This version of FastDIVA ' ...
        'is not yet implemented for the symmetric or block-deflation ' ...
        'separation scheme with the double nonstationarity model (L>1). ' ...
        'Now it continues with L set to 1 (the non-Gaussianity-based source model).']);
    L = 1;
end

%%%%%%%%%%% Blocks & sub-blocks

Nb = max(floor(N/T),1); % the length of blocks
if Nb ~= floor(N/T)
    Nb = floor(N/T);
    warning('FASTDIVA: The length of block(s) changed to %d; the number of blocks is %d.',floor(N/T),T);
end
N = Nb*T;

if N ~= size(x,2)
    x = x(:,1:N,:); % the last incomplete block is neglected
    warning('FASTDIVA: Data truncated to %d samples.',N);
end

Ns = Nb/L;
if Ns-floor(Nb/L)>0
    error('FASTDIVA: Nb/L must be integer!');
end

x = x - mean(x,2); % removing mean value from data

X = permute(reshape(permute(x,[2 1 3]), Nb, T, d, K),[3 1 4 2]); % blocks of data
Cb_in = mtimesx(X,X,'C')/Nb; % covariance matrices on blocks of input data
C_in = mean(Cb_in,4); % covariance matrices of entire input data

%%%%%%%%%%% Preconditioning (preprocessing)

if ~isfield(params,'precond') % pre-whitening
    params.precond = zeros(d,d,K);
    for k = 1:K
        params.precond(:,:,k) = sqrtm(inv(C_in(:,:,k)));
    end
elseif isempty(params.precond)
    params.precond = repmat(eye(d),1,1,K);
end

x = mtimesx(params.precond, x);
X = permute(reshape(permute(x,[2 1 3]), Nb, T, d, K),[3 1 4 2]);
Cb = mtimesx(X,X,'C')/Nb;
C = mean(Cb,4);   

%%%%%%%%%%% initialization
if ~isfield(params,'ini') % random initialization
    if params.initype == 'a'
        params.initype = 'w';
        warning('FastDIVA: Initialization type changed to ''w''.');
    end
    if realvalued
        Wini = randn(d,r,K);
    else 
        Wini = crandn(d,r,K);
    end
elseif strcmp(params.initype,'w') % initialization through the separating vectors/matrices
    if size(params.ini,1)~=d || size(params.ini,2)~=r || ~(size(params.ini,3)==K || size(params.ini,3)==1)
        error('FastDIVA: Invalid initialization.');
    elseif size(params.ini,3)==1 && K>1
        Wini = repmat(params.ini,1,1,K);
    else
        Wini = params.ini;
    end
elseif strcmp(params.initype,'a') % initialization through the mixing vectors/matrices
    Wini = zeros(d,r,K);
    if size(params.ini,4) == 1 % static initialization: MPDR
        for k = 1:K
            Wini(:,:,k) = C_in(:,:,k)\params.ini(:,:,k);
        end
    elseif size(params.ini,4) == T % dynamic initialization: LCMP
        for i = 1:r
            for k = 1:K
                A1 = squeeze(params.ini(:,i,k,:));
                aux = C_in(:,:,k)\A1;
                Wini(:,i,k) = aux*pinv(A1'*aux)*ones(T,1); % LCMP for all blocks
            end
        end
    else
        error('FastDIVA: Invalid initialization.');
    end
else
    error('FastDIVA: Unknown type of initialization.');
end

for k = 1:K
    Wini(:,:,k) = (params.precond(:,:,k)')\Wini(:,:,k);
end

% normalizing the initial output signal scales 
aux = sqrt(sum(mtimesx(C,Wini).*conj(Wini),1));
W = Wini./aux; 
Wt = zeros(d,r,K,T);

%%%%%%%%%%% THE ALGORITHM

%%% Non-gaussianity-based FastDIVA (version 1) for L==1

if L == 1

if params.approach == 'u' % One-Unit variant
    NumIt = zeros(1,r);
    for i = 1:r
        crit = 0;
        w = W(:,i,:);
        while crit < 1-epsilon && NumIt(i) < params.maxit
            NumIt(i) = NumIt(i) + 1;            
            a = mtimesx(Cb,w); 
            sigma2 = sum(conj(w).*a,1); % variance of SOI on blocks
            a = a./sigma2; % mixing vector
            wold = w./sqrt(mean(sigma2,4)); 
            soi = mtimesx(w,'C',X);
            sigma = sqrt(sigma2); 
            soin = soi./sigma; % block-normalized SOI 
            if realvalued
                [psi, psihpsi] = realnonln(soin, params.nonln);
            else
                [psi, psihpsi] = complexnonln(soin, params.nonln);
            end
            xpsi = (mtimesx(X,psi,'T')/Nb)./sigma;
            nu = mtimesx(w,'c',xpsi);
            rho = mean(psihpsi,2);
            gradw = a - xpsi./nu;
            gradw = sum(gradw,4); % gradient
            H2 = Cb./sigma2.*conj((nu-rho)./nu);
            H2 = sum(H2,4); % hessian
            for k = 1:K
                delta = -H2(:,:,k)\gradw(:,1,k);
                w(:,1,k) = w(:,1,k) + delta; % approx. Newton-Raphson iteration
            end
            w = w./sqrt(sum(mtimesx(C,w).*conj(w),1));
            crit = min(abs(sum(mtimesx(C,w).*conj(wold),1)),[],3);
        end
        Wt(:,i,:,:) = repmat(w,[1 1 1 T]);
    end
elseif params.approach == 'd' % Block-Deflation approach
    NumIt = zeros(1,r);
    P = zeros(d,d,K,T,r); % projection operators
    for i = 1:r
        crit = 0;        
        if i == 1
            Cbi = Cb;
            P(:,:,:,:,i) = repmat(eye(d),1,1,K,T);
        else
            v = W(1:d-i+2,i-1,:);
            a = mtimesx(Cbi,v);
            sigma2 = sum(conj(v).*a,1);
            a = a./sigma2;
            Pproj = eye(d-i+2)-mtimesx(a,v,'c');
            P(1:d-i+1,:,:,:,i) = mtimesx(Pproj(2:end,:,:,:),P(1:d-i+2,:,:,:,i-1));
            Cbi = mtimesx(mtimesx(Pproj(2:end,:,:,:),Cbi),Pproj(2:end,:,:,:),'C');
        end      
        w = mtimesx(P(1:d-i+1,:,:,1,i),W(:,i,:));
        while crit < 1-epsilon && NumIt(i) < params.maxit
            NumIt(i) = NumIt(i) + 1;            
            wt = mtimesx(P(1:d-i+1,:,:,:,i),'c',w);
            a = mtimesx(Cbi,w);
            sigma2 = sum(conj(w).*a,1);
            a = a./sigma2;
            soi = mtimesx(wt,'C',X);
            wold = w./sqrt(mean(sigma2,4));
            sigma = sqrt(sigma2); 
            soin = soi./sigma; % block-normalized SOI 
            if realvalued
                [psi, psihpsi] = realnonln(soin, params.nonln);
            else
                [psi, psihpsi] = complexnonln(soin, params.nonln);
            end
            xpsi = mtimesx(P(1:d-i+1,:,:,:,i),(mtimesx(X,psi,'T')/Nb)./sigma);
            nu = mtimesx(w,'c',xpsi);
            rho = mean(psihpsi,2);
            gradw = a - xpsi./nu;
            gradw = sum(gradw,4); % gradient
            H2 = Cbi./sigma2.*conj((nu-rho)./nu);
            H2 = sum(H2,4); % hessian
            for k = 1:K
                delta = -H2(:,:,k)\gradw(:,1,k);
                w(:,1,k) = w(:,1,k) + delta; % approx. Newton-Raphson iteration
            end
            w = w./sqrt(sum(mtimesx(mean(Cbi,4),w).*conj(w),1)); % normalization to unit output scale
            crit = min(abs(sum(mtimesx(mean(Cbi,4),w).*conj(wold),1)),[],3);
        end
        W(1:d-i+1,i,:) = w;
        Wt(:,i,:,:) = mtimesx(P(1:d-i+1,:,:,:,i),'c',w);
    end
elseif params.approach == 's' % Symmetric approach 
    NumIt = 0;
    crit = 0;
    while crit < 1-epsilon && NumIt < params.maxit
        NumIt = NumIt + 1;
        Wold = W;
        soi = mtimesx(W,'C',X);
        a = mtimesx(Cb,W);
        sigma2 = sum(conj(W).*a,1); % variance of SOI on blocks
        a = a./sigma2;
        sigma = sqrt(sigma2); 
        soin = soi./permute(sigma,[2 1 3 4]); % block-normalized SOI 
        if realvalued
            [psi, psihpsi] = realnonln(soin, params.nonln);
        else
            [psi, psihpsi] = complexnonln(soin, params.nonln);
        end
        nu = permute(mean(soin.*psi,2),[2 1 3 4]);
        rho = permute(mean(psihpsi,2),[2 1 3 4]);
        gradw = a - (mtimesx(X,psi,'T')/Nb)./nu./sigma;
        gradw = sum(gradw,4); % gradient
        for i = 1:r               
            H2 = Cb./sigma2(1,i,:,:).*conj((nu(1,i,:,:)-rho(1,i,:,:))./nu(1,i,:,:)); 
            H2 = sum(H2,4); % hessian
            for k = 1:K
                delta = -H2(:,:,k)\gradw(:,i,k);
                W(:,i,k) = W(:,i,k) + delta; 
            end
        end
        for k = 1:K
            W(:,:,k) = symdecor(W(:,:,k),C(:,:,k));
        end
        crit = min(min(abs(sum(mtimesx(C,W).*conj(Wold),1)),[],3));
    end
    Wt = repmat(W,[1 1 1 T]);
end

else % nonstationary model of SOI with L>1 (FastDIVA version 2)

X = permute(reshape(permute(x,[2 1 3]), Ns, L, T, d, K),[4 1 5 3 2]);
Cl = mtimesx(X,X,'C')/Ns; % covariances on sub-blocks
%Cb = mean(Cl,5);
%C = mean(Cb,4);   

if params.approach == 'u' % One-Unit variant
    NumIt = zeros(1,r);
    for i = 1:r
        crit = 0;
        w = W(:,i,:);
        while crit < 1-epsilon && NumIt(i) < params.maxit
            NumIt(i) = NumIt(i) + 1;
            al = mtimesx(Cl,w); 
            sigma2 = real(sum(conj(w).*al,1)); % variance of SOI on blocks
            a = sum(al,5)./sum(sigma2,5); % mixing vectors on blocks
            wold = w./sqrt(mean(mean(sigma2,4),5)); 
            %al = al./sigma2;
            soi = mtimesx(w,'C',X);
            sigma = sqrt(sigma2); 
            soin = soi./sigma;
            if gauss
                if strcmp(params.nonln,'gauss')
                    ss = permute(soin,[3 2 1 4 5]);
                    Ck = mtimesx(ss,ss,'C')/Ns;   
                    if realvalued %%%
                        Pk = zeros(size(Ck));%%%
                    else
                        Pk = mtimesx(ss,ss,'T')/Ns;%%%
                    end %%%
                    iP = zeros(K,K,1,T,L); %%%
                    P = zeros(K,K,1,T,L);
                    R = zeros(K,K,1,T,L);
                    rho = zeros(1,1,K,T,L);            
                    for t = 1:T
                        for l = 1:L
                            if K>10, Ck(:,:,1,t,l) = Ck(:,:,1,t,l) + 0.1*eye(K); end
                            aux = Ck(:,:,1,t,l)\Pk(:,:,1,t,l);
                            R(:,:,1,t,l) = aux';
                            P(:,:,1,t,l) = conj(Ck(:,:,1,t,l))-Pk(:,:,1,t,l)'*aux;
                            iP(:,:,1,t,l) = inv(P(:,:,1,t,l));
                            rho(1,1,:,t,l) = diag(iP(1:K,1:K,1,t,l));
                        end
                    end
                    aux = mtimesx(iP,R);
                    aux = (aux + permute(aux,[2 1 3 4 5]))/2;
                    psi = mtimesx(iP,conj(ss))-mtimesx(aux,ss);
                    psi = permute(psi,[3, 2, 1, 4, 5]);
                    nu = 1;
                elseif  strcmp(params.nonln,'gausstri')      
                    Ck = mean(soin(:,:,1:K-1,:,:).*conj(soin(:,:,2:K,:,:)),2);
                    Ck(abs(Ck)>0.4) = 0.4*sign(Ck(abs(Ck)>0.4));
                    Ck2 = real(Ck.*conj(Ck));
                    theta = zeros(1,1,K+1,T,L);
                    phi = zeros(1,1,K+1,T,L);
                    theta(1,1,1:2,:,:) = 1;
                    phi(1,1,K:K+1,:,:) = 1;
                    for k = 2:K
                        theta(1,1,k+1,:,:) = theta(1,1,k,:,:) - Ck2(1,1,k-1,:,:).*theta(1,1,k-1,:,:);
                        phi(1,1,K+1-k,:,:) = phi(1,1,K+2-k,:,:) - Ck2(1,1,K+1-k,:,:).*phi(1,1,K+3-k,:,:);
                    end
                    rho = theta(1,1,1:K,:,:).*phi(1,1,2:K+1,:,:)./theta(1,1,K+1,:,:);
                    iP = zeros(K,K,1,T,L);
                    for t = 1:T
                        for l = 1:L
                            iP(:,:,1,t,l) = diag(squeeze(rho(1,1,:,t,l)));
                            for k = 1:min(K-1,5)
                                aux = theta(1,1,1:K-k,t,l).*phi(1,1,k+2:K+1,t,l)/theta(1,1,K+1,t,l);
                                for j = 1:k
                                    aux = aux.*(-Ck(1,1,j:K-k+j-1,t,l));
                                end
                                iP(:,:,1,t,l) = iP(:,:,1,t,l) + diag(squeeze(aux),k) + diag(squeeze(conj(aux)),-k);
                            end
%                            Pk(:,:,1,t,l) = diag(diag(Pk(:,:,1,t,l)))/8; % + diag(aux,1) + diag(aux,-1);
                        end
                    end
%                    aux = mtimesx(iP,R);
%                    aux = (aux + permute(aux,[2 1 3 4 5]))/2;
                    iP = conj(iP); % because of the definition of P, which is here P=conj(\Sigma)
                    ss = permute(soin,[3 2 1 4 5]);
                    psi = mtimesx(iP,conj(ss)); %-mtimesx(aux,ss);
                    psi = permute(psi,[3, 2, 1, 4, 5]);                    
                    nu = mean(soin.*psi,2);
                end                
            else % non-Gauss
                if realvalued
                    [psi, psihpsi] = realnonln(soin, params.nonln);
                else
                    [psi, psihpsi] = complexnonln(soin, params.nonln);
                end
                nu = mean(soin.*psi,2);
                rho = mean(psihpsi,2);
            end
            
            grad = a - mean(mean(psi.*X,2)./sigma./nu,5);
            if fasthessian % FastDIVA Hessian
                H = Cb./mean(sigma2,5)-mean(rho.*Cl./sigma2./conj(nu),5);
            else % QuickIVE Hessian
                H = -mean(rho.*Cl./sigma2./conj(nu),5);
            end
            grad = mean(grad,4);
            H = mean(H,4);    
            for k = 1:K
                delta = -H(:,:,k)\grad(:,1,k);
                w(:,1,k) = w(:,1,k) + delta; % approx. Newton-Raphson iteration
            end
            w = w./sqrt(sum(mtimesx(C,w).*conj(w),1));
            crit = min(abs(sum(mtimesx(C,w).*conj(wold),1)),[],3);            
        end
        Wt(:,i,:,:) = repmat(w,[1 1 1 T]);
    end
end % block-deflation and symmetric variants TBD

end

%%%%%%%%%%% Scaling of outputs
X = permute(reshape(permute(x,[2 1 3]), Nb, T, d, K),[3 1 4 2]);
Sn = mtimesx(Wt,'c',X);    
Wt = mtimesx(params.precond,'C',Wt);
aux = mtimesx(Cb_in,Wt);
sigma2 = real(sum(conj(Wt).*aux,1));
At = aux./sigma2;

if nargout > 4 % theoretical mean ISR achieved by one-unit FastDIVA
    sigma = sqrt(sigma2);
    soin = Sn./permute(sigma,[2 1 3 4]); % block-normalized SOI
    if realvalued
        [psi, psihpsi] = realnonln(soin, params.nonln);
    else
        [psi, psihpsi] = complexnonln(soin, params.nonln);
    end
    nu = permute(mean(soin.*psi,2),[2 1 3 4]);
    rho = permute(mean(psihpsi,2),[2 1 3 4]);
    phi = permute(mean(conj(psi).*psi,2),[2 1 3 4]);
    aux = permute(At,[1 5 3 4 2]);
    Idm1 = repmat(eye(d-1),1,1,K,T,r);
    B = [aux(2:end,1,:,:,:), -aux(1,1,:,:,:).*Idm1];
    Cz = mtimesx(B,mtimesx(Cb_in,B,'C'));
    ISRest1u = zeros(r,K);
    for k = 1:K
        for i = 1:r
            R = inv(mean(Cz(:,:,k,:,i).*(nu(1,i,k,:)-rho(1,i,k,:))./sigma2(1,i,k,:)./nu(1,i,k,:),4));
            S = mean(Cz(:,:,k,:,i).*(phi(1,i,k,:)-abs(nu(1,i,k,:)).^2)./sigma2(1,i,k,:)./(abs(nu(1,i,k,:)).^2),4);
            ISRest1u(i,k) = real(trace((mean(Cz(:,:,k,:,i),4)/mean(sigma2(1,i,k,:),4))*R*S*R')/N);
        end
    end
end

if params.scaling == 'n' % normalized signals
    S = permute(reshape(permute(Sn,[2 4 1 3]),[N r K]),[2 1 3]);
elseif params.scaling == 'i' % least-squares sources' images on sensors
    S = zeros(d,N,K,r);
    for i = 1:r
        aux = mtimesx(At(:,i,:,:),Sn(i,:,:,:));
        S(:,:,:,i) = reshape(permute(aux,[1 2 4 3]),d,Nb*T,K);
    end
elseif params.scaling == '1' % least-squares sources' images on the first sensor
    S = zeros(r,N,K);
    for i = 1:r
        aux = mtimesx(At(1,i,:,:),Sn(i,:,:,:));
        S(i,:,:) = reshape(permute(aux,[1 2 4 3]),1,Nb*T,K);
        Wt(:,i,:,:) = conj(At(1,i,:,:)).*Wt(:,i,:,:);
        At(:,i,:,:) = At(:,i,:,:)./At(1,i,:,:);
    end
end
end

%%%%%%%%%%% helping functions

function [psi, psipsi] = realnonln(s,nonln)
    if strcmp(nonln,'sign')
        if size(s,3)==1, error('FastDIVA: Nonlinearity "sign" cannot be used for the real-valued ICA/ICE.'); end
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

function W = symdecor(M,C)
%fast symmetric orthogonalization
    if nargin == 2
        [V, D] = eig(M'*C*M);
    else
        [V, D] = eig(M'*M);
    end
    W = M*(V./sqrt(diag(D)'))*V';
end

function y = newmtimesx(varargin)
    if nargin==2
        y = pagemtimes(varargin{1},'none',varargin{2},'none');
    elseif nargin==3
        if ischar(varargin{2})
            y = pagemtimes(varargin{1},varargin{2},varargin{3},'none');
        else
            y = pagemtimes(varargin{1},'none',varargin{2},varargin{3});
        end
    else
        y = pagemtimes(varargin{1},varargin{2},varargin{3},varargin{4});
    end
end