function [w, lambda, soi, NumIt] = oglps_single(x, lambdaini, MaxIt, nonln, inform, v)
% Independent Component Extraction by Orthogonally-Constrained Gradient
% algorithm - optimizing the linear phase-shift mixing vector
%
% The algorithm is designed to extract an independent component. It depends
% on the initialization which component is being extracted. 
%
% USAGE: [w, a, shat, NumIt] = oglps_a(x, mu, aini, MaxIt, nonln, precond);
% 
% inputs (real or complex-valued):
% 
% x ... d x N matrix of complex-valued observed mixed signals; d sensors; N samples
% mu ... step size (default = 0.1)
% lambdaini ... real initial value for the mixing vector (default: lambdaini = 0)
% MaxIt ... maximum number of iterations (default = 1000);
% nonln ... nonlinearity in place of score function (default = rati)
%
% outputs (real or complex-valued):
%
% w ... d x 1 complex-valued separating vector estimate
% lambda ... real phase-shift mixing vector parameter estimate
% soi ... 1 x N vector containing the extracted complex-valued signal-of-interest
% NumIt ... number of iterations
%
% Coded by Zbynek Koldovsky, last change November 2023
%
% Reference:
%
% Z. Koldovský and P. Tichavský, "Gradient Algorithms for Complex Non-Gaussian 
% Independent Component/Vector Extraction, Question of Convergence," 
% IEEE Trans. on Signal Processing, 2018.
%
% Author(s): Zbynìk Koldovský
% Technical University of Liberec
% Studentská 1402/2, LIBEREC
% Czech Republic
%
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

[d, N, K] = size(x);
iotta = 0.001;

if isreal(x), error('This algorithm is only for complex-valued signals and system parameters!'); end

if nargin < 4
    nonln = 'rati';
end

if nargin < 3
    MaxIt = 1000;
end

if nargin < 2
    lambdaini = 0;
end

if nargin < 6
    v = repmat((0:d-1)',1,1,K);
end
    
lambda = lambdaini;
 
Cx = pagemtimes(x,'none',x,'ctranspose')/N + 0.01*eye(d);
%Cxinv = pageinv(Cx);

if nargin > 4
    if size(inform,1)>1
        inform = permute(inform,[3 2 1]);
    end
    alphaweights = 1./(iotta+abs(inform));
else
    alphaweights = ones(1,N,K);
end
alphaweights = alphaweights./mean(alphaweights,2);
Cw = pagemtimes(x.*alphaweights, 'none', x, 'ctranspose')/N;
Cwinv = pageinv(Cw);
%R = Cwinv*Cx;

%w = repmat(eye(d,1),1,1,K);
crit = 0.5;
NumIt = 0;
%mu = 0.01;

%Id = eye(d);
%e1 = Id(:,1);
%E = Id(:,2:end);

a = exp(1i*(lambda.*v));

while 1-crit > 1e-9 && NumIt < MaxIt
%while crit > 1e-7 && NumIt < MaxIt
    NumIt = NumIt + 1;
    
    aold = a;
    w = pagemtimes(Cwinv, 'none', a, 'none'); % MVDR constraint
    sigmaw2 = real(1./pagemtimes(a, 'ctranspose', w, 'none'));
    w = w.*sigmaw2;
    sigmax2 = real(pagemtimes(pagemtimes(w, 'ctranspose', Cx, 'none'), 'none', w, 'none'));

    soi = pagemtimes(w, 'ctranspose', x, 'none');
    sigmax = sqrt(sigmax2); 
    soin = soi./sigmax; % normalized SOI 
    [psi, psihpsi] = complexnonln(soin, nonln);
    xpsi = (pagemtimes(x, 'none', psi, 'transpose')./sigmax)/N;
    nu = pagemtimes(w, 'ctranspose', xpsi, 'none');
    rho = mean(psihpsi,2);
    %xi = mean(psihpsi.*soin.*conj(soin));
    %eta = mean(psipsi.*soin.*soin);

    % a_orth = pagemtimes(Cx,'none',w,'none');
    % a_orth = a_orth./pagemtimes(a_orth,'ctranspose',w,'none');

    % B = [a(2:end) -eye(d-1)*a(1)];
    % Czinv = inv(B*Cx*B');
    % q = B*Cx*w;
    % grad_a = sigmaw2*Cwinv*(a - xpsi/nu) + [a(2:end)'*Czinv*q;-conj(a(1))*Czinv*q];
    aOG = pagemtimes(Cx, 'none', w, 'none')./sigmax2;
    grad_a = sigmaw2.*pagemtimes(Cwinv, 'none', ...
        aOG - xpsi./nu, 'none');


    %grad_w = a - xpsi/nu;
    av = a.*v;
    grad_lambda = -2*imag(pagemtimes(grad_a, 'ctranspose', av, 'none')); %grad_a.'*(1i*v.*a) - grad_a'*(1i*v.*conj(a));
    
    dadl = 1i*av;
    %c1 = (nu-rho)/nu/sigmax2;
    % c3 = (xi-eta-nu)/2/nu;
    % c2 = -sigma2*c1-c3;
    % H1 = conj(c3*(a*a.'));
    % if nargin>4
    %     H2 = c1*Cxweighted.';
    % %else
    % %    H2 = (c1*Cx + c2*(a*a')).';
    % end
    %wOG = sigmaw2*Cwinv*aOG;

    H2 = ((nu-rho)./nu).*(sigmaw2.^2).*pagemtimes(pagemtimes(Cwinv, 'none',...
        Cx./sigmax2 - pagemtimes(aOG, 'none', aOG, 'ctranspose'), 'none'), 'none',...
        Cwinv, 'none');
    %H2 = sigmaw2*Cwinv*sigmaw2/sigmax2*(Cx*Cwinv-a*w')*(nu-rho)/nu;
    dgadl = pagemtimes(H2, 'ctranspose', dadl, 'none'); % + conj(H1.'*dwdl);
    %ds2dl = 2*sigma2*imag(w'*av);
    d2ldl2 = -2*imag(pagemtimes(dgadl, 'ctranspose', av, 'none'));

    % H = sigmaw2*(nu-rho)/nu*(Cwinv);
    % d2ldl2 = -2*av'*H*av;

    %d2ldl2 = 2*sigmaw2*(av'*((1-(rho./nu))/sigmax2))*(Cwinv*av);

    %lambda = lambda - grad_lambda/(1-(rho./nu)).*(sigmaw2./sigmax2)/2;
   % ratio = sigmaw2/sigmax2;
   % d2ldl2 = 2*sigmaw2*(av'*(eye(d) - rho/nu*ratio))*(Cwinv*av);
    lambda = lambda - mean(grad_lambda,3)/mean(d2ldl2,3);

    %lambda = lambda + sigmax2*grad_lambda/sigmaw2;

    %a = a + mu*grad_a;
    a = exp(1i*(lambda.*v));
    %crit = norm(grad_lambda);
    crit = min(abs(sum(a.*conj(aold),1))./sqrt(sum(a.*conj(a),1).*sum(aold.*conj(aold),1)),[],3);
    %disp(crit)
end

%a = exp(1i*lambda.*v);
w = pagemtimes(Cwinv, 'none', a, 'none'); % MVDR constraint
sigmaw2 = real(1./pagemtimes(a, 'ctranspose', w, 'none'));
w = w.*sigmaw2;
soi = pagemtimes(w, 'ctranspose', x, 'none');


end


function [psi, psipsih, psipsi] = complexnonln(s,nonln)
    if strcmp(nonln,'sign')
        sp2 = s.*conj(s);
        aux = 1./sqrt(sum(sp2,3));
        psi = conj(s).*aux;
        psipsih = aux.*(1-psi.*conj(psi)/2);
    elseif strcmp(nonln,'rati')
        sp2 = s.*conj(s);
        aux = 1./(1+sum(sp2,3));
        psi = conj(s).*aux;
        psipsih = aux - psi.*conj(psi);   
        psipsi = -psi.^2;
    end
end