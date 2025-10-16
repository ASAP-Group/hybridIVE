clear *
close all

% This script performs the simulation described in Section V.A of [a]; more
% methods are compared here than in the original article

% [a] Z. Koldovsky, J. Malek, M. Vratny, T. Vrbova, J. Cmejla, and S.
% O'Regan, "From Informed Independent Vector Extraction to Hybrid 
% Architectures for Target Source Extraction", Oct. 2025 (this article)
%
% [b] Z. Koldovský, J. Málek, J. Čmejla, M. Vrátný, and W. Kellermann, 
% “Fast Algorithms for Informed Independent Component/Vector Extraction,” 
% EURASIP Journal on Advances in Signal Processing, vol. x, pp. xx-xx, 
% accepted, Sept 2025.
%
% [c] Z. Koldovsky, V. Kautsky, P. Tichavsky, J. Cmejla, and J. Malek,
% “Dynamic independent component/vector analysis: Time-variant linear
% mixtures separable by time-invariant beamformers,” IEEE Transactions
% on Signal Processing, vol. 69, pp. 2158–2173, 2021.
%
% [d] J. Jansky, Z. Koldovsky, J. Malek, T. Kounovsky, and J. Cmejla,
% “Auxiliary function-based algorithm for blind extraction of a moving
% speaker,” EURASIP Journal on Audio, Speech, and Music Processing,
% vol. 2022, p. 1, Jan 2022.


domain = 'complex';
ndist = 'laplace'; % the distribution of interference signals

%%%%%%%%%% Data

N = 200; % number of samples
T = 1; % # blocks (ICE = 1, CSV > 1)
L = 10; % # subblocks
d = 6; % # sources
r = d; % # channels
K = 6; % # mixtures (ICE = 1, IVE > 1 )
minvar = sqrt(0.1);
maxvar = 10;
SIRinlow = -5;
SIRinhigh = -5;
circularitycoef = 0; 
delta2 = 0.1;
xi = 0.4;
tau = 2;
epsilon2 = 0.5;
numMethods = 13;

%%%%%%%%%% Simulation
ntrials = 1000; %number of trials
parameters = [0.01 0.1 0.3 0.5 0.8 1]; % values of the tested parameter

%%%%%%%%%%% Outputs
itertime = zeros(ntrials,length(parameters),K,numMethods);
iterations = zeros(ntrials,length(parameters),K,numMethods);
oISR = zeros(ntrials,length(parameters),K,numMethods);
iISR = zeros(ntrials,length(parameters),K);


for ind_param = 1:length(parameters)

epsilon2 = parameters(ind_param);
Nb = N/T;
Ns = Nb/L;


variances = zeros(d,N,K);
subvariances = zeros(d,Nb,K);
for k = 1:K   % variance profiles of signals
    variances(1,:,k) = kron(sin((1:T)/(T+1)*pi).^(tau/2),ones(1,Nb));
    variances(2:end,:,k) = kron(sqrt(minvar + (maxvar-minvar)*rand(d-1,T)),ones(1,Nb));
    
    subvariances(1,:,k) = kron(abs(sin((1:L)/(L+1)*pi)).^(tau/2),ones(1,Ns));
    subvariances(2:end,:,k) = ones(d-1,Nb);
end

variances = variances.*repmat(subvariances,[1 T 1]);

   
parfor trial = 1:ntrials

if strcmp(domain,'real')
    U = orth(randn(K));
elseif strcmp(domain,'complex')
    U = orth(crandn(K));
end

trialresults = zeros(4,K,numMethods);

disp(['parameter index:' num2str(ind_param) ' trial:' num2str(trial)]);

%% Sources
s = zeros(d,N,K);

if strcmp(domain,'real')
    aux = gengau2(alpha,K,N);
    s(1,:,:) = permute(U*(permute(variances(1,:,:),[3 2 1]).*aux),[3 2 1]);
    if strcmp(ndist,'normal')
        s(2:d,:,:) = variances(2:d,:,:).*randn(d-1,N,K); 
    else
        s(2:d,:,:) = variances(2:d,:,:).*laplace(d-1,N,K); 
    end
elseif strcmp(domain,'complex')
    aux = zeros(K,N);
    for i = 1:K
       aux(i,:) = cggd_rand(xi, 1, N, circularitycoef); 
    end
    s(1,:,:) = permute(U*(permute(variances(1,:,:),[3 2 1]).*aux),[3 2 1]);
    if strcmp(ndist,'normal')
        s(2:d,:,:) = variances(2:d,:,:).*crandn(d-1,N,K); 
    else
        s(2:d,:,:) = variances(2:d,:,:).*claplace(d-1,N,K); 
    end
end

%% initial SIR setting
powers = squeeze(sum(s.*conj(s),2));
powers = [powers(1,:); mean(powers(2:end,:),1)];
inputISR = powers(2,:)./powers(1,:);
setSR = SIRinlow + (SIRinhigh-SIRinlow)*rand(1,K); 
gain = sqrt(inputISR.*10.^(setSR/10)); 
s(1,:,:) = permute(gain,[1 3 2]).*s(1,:,:);

%% (De-)Mixing matrices
if strcmp(domain,'real')
    Wtrue = 1+rand(r,d,K,T);
elseif strcmp(domain,'complex')
    Wtrue = 1+rand(r,d,K,T)+1i*rand(r,d,K,T);
end
Wtrue(1:1,:,:,:) = repmat(Wtrue(1:1,:,:,1),1,1,1,T); 
Atrue = pageinv(Wtrue);
x = zeros(r,N,K);
noise = zeros(r,N,K);

IVEini = zeros(d,K);
IVEwini = zeros(d,K);
priorinformationIVE = zeros(K,N);
trial_iISR = zeros(K,1);



for k = 1:K % ICE tests for each mixture

% Observations
A = squeeze(Atrue(:,:,k,:)); 
sk = s(:,:,k);
xk = zeros(r,N);
for t = 1:T
    xk(:,(t-1)*Nb+1:t*Nb) = A(:,:,t)*sk(:,(t-1)*Nb+1:t*Nb);
    noise(:,(t-1)*Nb+1:t*Nb,k) = A(:,2:end,t)*sk(2:end,(t-1)*Nb+1:t*Nb);
end

x(:,:,k) = xk;

powers = real(mean(sk.*conj(sk),2));
trial_iISR(k) = mean(powers(2:end))./powers(1);

%% Testing

% initialization
if strcmp(domain,'real')
    difference = randn(r,1);
else
    difference = crandn(r,1);
end
wtrue = Wtrue(1,:,k,1)';
difference = difference-(wtrue'*difference)/(wtrue'*wtrue)*wtrue;
difference = difference/norm(difference)*sqrt(delta2);
wini = wtrue + difference;
Cx = xk*xk'/N;
aux = Cx*wini;
aini = aux/(wini'*aux);
Wini = [wini'; [aini(2:end) -aini(1)*eye(d-1)]];

priorinformation = sqrt(1-epsilon2)*sk(1,:)/sqrt(mean(abs(sk(1,:)).^2)) + sqrt(epsilon2)*crandn(size(sk(1,:)));
priorinformation = kron(sqrt(1-epsilon2)*mean(reshape(abs(priorinformation).^2,[Ns L]),1) + sqrt(epsilon2)*rand(1,L),ones(1,Ns));

for method = 1:8 % ICE/ICA - each mixture is processed separately
    tic
    w = wini;
    NumIt = 0;
    switch method
        case 1 % initialization
            w = wini;
        case 2 % iFICA (informed FastIVE - implementation from [b])
            [w, a, ~, NumIt] = iFICA(xk, wini, priorinformation.^2, 'rati');
        case 3 % iFICA-G (compared in [b]]
            %[w, a, ~, NumIt] = iFICAGauss(xk, L, wini, priorinformation.^2);
        case 4 % piloted AuxIVE [d]
            [~, out] = auxogive_final(xk, struct('MaxIt', 100, 'BlockSize', Nb, 'wini', wini, 'P', abs(priorinformation).^2));
            w = out.w;
            NumIt = out.NumIt;
        case 5 % blind FastIVA nongauss (implementation from [c])
            [w, ~, ~, NumIt] = fastdiva(xk, struct('ini', wini, 'initype', 'w', 'T', T, 'L', 1, 'nonln', 'rati'));
            w = w(:,1,1,1);
        case 6 % FastDIVA gauss nonstat (compared in [b]]
            %[w, ~, ~, NumIt] = fastdiva(xk, struct('ini', wini, 'initype', 'w', 'T', T, 'L', L, 'nonln', 'gauss'));
            %w = w(:,1,1,1);
        case 7 % approximate MVDR
            Cw = pagemtimes((xk./(abs(priorinformation).^2+0.001)), 'none', xk, 'ctranspose')/N;
            w = Cw\aini;
            w = w/(aini'*w);
        case 8 % iFastIVE
            [w, a, ~, NumIt] = ifastive(xk, aini, priorinformation.^2, 'rati', 0.001);
        case 9 %    
        case 10 % 
        case 12 %
    end
    resultingISR = ISR_CSV(w,xk,noise(:,:,k));
    trialresults(1,k,method) = toc;
    trialresults(2,k,method) = NumIt;
    trialresults(3,k,method) = resultingISR;  
end

IVEini(:,k) = aini;
IVEwini(:,k) = wini;
priorinformationIVE(k,:) = priorinformation;
end

for method = 10:13 % IVE/IVA - all K mixtures processed jointly
    tic
    switch method
        case 10 % (informed FastIVE - implementation from [b])
            [w, a, ~, NumIt] = iFICA(x, permute(IVEwini,[1 3 2]), priorinformationIVE.^2, 'rati');
        case 11 % iFastIVE
            [w, a, ~, NumIt] = ifastive(x, permute(IVEini,[1 3 2]), priorinformationIVE.^2, 'rati', 0.001);
        case 12 % p-AuxIVA [d]
            [~, out] = auxogive_final(x, struct('MaxIt', 100, 'BlockSize', Nb, 'wini', IVEwini, 'P', abs(priorinformationIVE).^2));
            w = permute(out.w,[1 2 4 3]);
            NumIt = out.NumIt;
        case 13 % blind FastIVE (implementation from [c])
            [w, ~, ~, NumIt] = fastdiva(x, struct('approach','u','ini',permute(IVEwini,[1 3 2]),'initype','w','T', T, 'L', 1, 'maxit', 100, 'nonln', 'rati'));
            %[w, ~, ~, NumIt] = fastdiva(x, struct('approach','u','ini',permute(IVEwini,[1 3 2]),'initype','w','T', T, 'L', L, 'maxit', 100, 'nonln', 'gauss'));
            w = w(:,:,:,1);
    end
    trialresults(1,:,method) = toc;
    trialresults(2,:,method) = NumIt;
    trialresults(3,:,method) = ISR_CSV(w,x,noise);
end


itertime(trial,ind_param,:,:) = permute(trialresults(1,:,:),[1 4 2 3]);
iterations(trial,ind_param,:,:) = permute(trialresults(2,:,:),[1 4 2 3]);
oISR(trial,ind_param,:,:) = permute(trialresults(3,:,:),[4 1 2 3]);
iISR(trial,ind_param,:) = trial_iISR;
end

end

%% Vizualize

order = [1 7 5 6 13 2 10 11 3 4 12]; % 5 6 7 8 9 10 11 12 13];
legenda = {'initial','aMVDR','FICA','FICA-G', 'FIVA', 'iFICA', 'iFIVA', 'iFastIVE', 'iFICA-G', ... %'iFIVA-G',...
    'p-AuxICA', 'p-AuxIVA'};
lines = {':',':','--','--','--','-','-','-','-','-.','-.',':',':','-'};
marks = {'none','x','^','s','v','^','v','o','s','^','v','x','o','none'};
cols = {'k', 'b', [0 0.5 0.8], [0 0.5 0.8], [0 0.5 0.8], [1 0.2 0], [1 0.2 0], [1 0.2 0], [0 0.6 0],  ...
    [0.7 0.5 0],[0.7 0.5 0]}; %[0.6 0 0.6], [0.7 0.5 0],'k','k',[0.7 0.7 0.7],'k'};
widths = {3,3,2,2,2,2,2,2,2,2,2,2,2,2,2};

f = figure; f.Position(2:4) = [400 600 500];
subplot('Position',[0.15 0.15 0.33 0.65])
%aux = double(10*log10(oISR(:,:,:,order))<8 & 10*log10(oISR(:,:,:,order))>-8); 
aux = 10*log10(oISR(:,:,:,order))<-3; 
h = plot(parameters,squeeze(mean(mean(aux,1,'omitnan'),3,'omitnan')*100));
%axis([0.001 5 10 100])
set(gca,'FontSize',14)
for k = 1:length(h)
    h(k).LineStyle = lines{k};
    h(k).LineWidth = widths{k};
    h(k).Color = cols{k};
    h(k).Marker = marks{k};
    h(k).MarkerSize = 8;
end
grid on
%title(['Number of convergences out of ' num2str(ntrials) ' trials'])
%title(['Global convergence in ' num2str(ntrials) ' trials'],'FontSize',16)
ylabel('success rate [%]','FontSize',16)
xlabel('\epsilon^2','FontSize',16)

subplot('Position',[0.6 0.15 0.33 0.65])
%aux = double(10*log10(oISR(:,:,:,order))<8 & 10*log10(oISR(:,:,:,order))>-8); 
aux = oISR(:,:,:,order);
aux(10*log10(aux)>-3) = nan; 
h = plot(parameters,-10*log10(squeeze(mean(mean(aux,1,'omitnan'),3,'omitnan'))));
%axis([0.001 5 10 100])
set(gca,'FontSize',14)
for k = 1:length(h)
    h(k).LineStyle = lines{k};
    h(k).LineWidth = widths{k};
    h(k).Color = cols{k};
    h(k).Marker = marks{k};
    h(k).MarkerSize = 8;
end
grid on
%title(['Number of convergences out of ' num2str(ntrials) ' trials'])
%title(['Global convergence in ' num2str(ntrials) ' trials'],'FontSize',16)
ylabel('average successful SIR [dB]','FontSize',16)
xlabel('\epsilon^2','FontSize',16)
legend(legenda,'Position',[0.13 0.83 0.75 0.1],'Orientation','horizontal','NumColumns',5);

