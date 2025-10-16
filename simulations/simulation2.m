close all
clear *

% This script performs the simulation described in Section V.B of [a]

% [a] Z. Koldovsky, J. Malek, M. Vratny, T. Vrbova, J. Cmejla, and S.
% O'Regan, "From Informed Independent Vector Extraction to Hybrid 
% Architectures for Target Source Extraction", Oct. 2025 (this article)


%%%%%%%%%% Data
N = 30; % # samples
d = 4; % dimension 
K = 5; % # modes (the number of models, ICA... = 1, IVA... > 1 )
L = 1; % # intervals for non-stationary model of signals
maxIter = 100;
numMethods = 15;
SIRinlow = -10;
SIRinhigh = 10;
ndist = 'laplace';
nonln = 'rati';
circularitycoef = 0; 
iniperturbation = 0.1;
alpha = 0.4;
lambdatrue = 0.5;
minvar = sqrt(0.1);
maxvar = 10;
epsilon2 = 0.4;
modelperturbation = 0; %0.0001;

ntrials = 1000;

parameters = [10 20 30 50 80 100 200 500 1000];

itertime = zeros(ntrials,length(parameters),K,numMethods);
iterations = zeros(ntrials,length(parameters),K,numMethods);
oISR = zeros(ntrials,length(parameters),K,numMethods);
iISR = zeros(ntrials,length(parameters),K);
OKtrial = zeros(ntrials,length(parameters),K,numMethods);


for ind_param = 1:length(parameters)

N = parameters(ind_param);
Ns = N/L;

parfor trial = 1:ntrials
    
trialresults = zeros(4,K,numMethods);

disp(['parameter index:' num2str(ind_param) ' trial:' num2str(trial)]);

s = zeros(d,N,K);

U = orth(crandn(K));
aux = zeros(K,N);
for i = 1:K
   aux(i,:) = cggd_rand(alpha, 1, N, circularitycoef); 
end
s(1,:,:) = permute(U*aux,[3 2 1]);
if strcmp(ndist,'normal')
    s(2:d,:,:) = crandn(d-1,N,K); 
else
    s(2:d,:,:) = claplace(d-1,N,K); 
end

v = (0:d-1)';
Atrue = [repmat(exp(1i*lambdatrue.*v),[1 1 K]) crandn(d,d-1,K)];
Atrue = Atrue + modelperturbation*crandn(d,d,K);
Wtrue = pageinv(Atrue);
wtrue = permute(conj(Wtrue(1,:,:,:)),[2 1 3 4]);

powers = squeeze(sum(s.*conj(s),2));
powers = [powers(1,:); mean(powers(2:end,:),1)];
inputISR = powers(2,:)./powers(1,:);
setSR = SIRinlow + (SIRinhigh-SIRinlow)*rand(1,K); %double(rand(1,M)>0.5);
gain = sqrt(inputISR.*10.^(setSR/10)); 
s(1,:,:) = permute(gain,[1 3 2]).*s(1,:,:);
powers = squeeze(sum(s.*conj(s),2));
powers = [powers(1,:); mean(powers(2:end,:),1)];
iISR(trial,ind_param,:) = powers(2,:)./powers(1,:);

x = zeros(d,N,K);
noise = zeros(d,N,K);

IVEini = zeros(1,K);
IVEaini = zeros(d,K);
priorinformationIVE = zeros(K,N);

trial_iISR = zeros(K,1);

for k = 1:K
%% Generating signals    
A = Atrue(:,:,k); 
sk = s(:,:,k);
xk = A*sk; 
noise(:,:,k) = A(:,2:end)*sk(2:end,:);

x(:,:,k) = xk;

Cx = xk*xk'/N;

powers = mean(sk.*conj(sk),2);
trial_iISR(k) = mean(powers(2:end))./powers(1);

%% Testing

% initialization
lambdaini = lambdatrue + sqrt(iniperturbation)*randn;
aini = exp(1i*lambdaini*v);
aux = Cx\aini;
wini = aux/(aini'*aux);

priorinformation = sqrt(1-epsilon2)*sk(1,:)/sqrt(mean(abs(sk(1,:)).^2)) + sqrt(epsilon2)*crandn(size(sk(1,:)));


for method = 1:1 % We do not consider ICE/ICA in this simulation, only initialization is evaluated
    tic
    w = zeros(d,1);
    NumIt = 0;
    switch method
        case 1 % ini
            w = wini;
            a = lambdaini;
    end
    resultingISR = ISR_CSV(w,xk,noise(:,:,k));
    trialresults(1,k,method) = toc/NumIt;
    trialresults(2,k,method) = NumIt;
    trialresults(3,k,method) = resultingISR;
end

IVEini(1,k) = lambdaini;
IVEaini(:,k) = aini;
priorinformationIVE(k,:) = priorinformation;
end



for method = 10:numMethods % IVE/IVA
    tic
    w = zeros(d,K);
    NumIt = 0;
    switch method
        case 10 % PSIVE
            [w, a, ~, NumIt] = ipsive(x, permute(IVEini,[3 1 2]), 100, nonln);
        case 11 % iPSIVE
            [w, a, ~, NumIt] = ipsive(x, permute(IVEini,[3 1 2]), 100, nonln, priorinformationIVE);
        case 12 % iFastIVE
            [w, a, ~, NumIt] = ifastive(x, permute(IVEaini,[1 3 2]), priorinformationIVE, 'rati', 0.001);
        case 13 % CaponIVE
            [w, a, ~, NumIt] = icaponive(x, IVEini(1), 100, nonln);  
        case 14 % iCaponIVE
            [w, a, ~, NumIt] = icaponive(x, IVEini(1), 100, nonln, priorinformationIVE);
        case 15 % FastIVE
            [w, a, ~, NumIt] = ifastive(x, permute(IVEaini,[1 3 2]));
    end
    trialresults(1,:,method) = toc;
    trialresults(2,:,method) = NumIt;
    trialresults(3,:,method) = ISR_CSV(w,x,noise);
    if sum(method==[10 11 14])>0
        trialresults(4,:,method) = squeeze(a);
    end
end

itertime(trial,ind_param,:,:) = permute(trialresults(1,:,:),[1 4 2 3]);
iterations(trial,ind_param,:,:) = permute(trialresults(2,:,:),[1 4 2 3]);
oISR(trial,ind_param,:,:) = permute(trialresults(3,:,:),[1 4 2 3]);
OKtrial(trial,ind_param,:,:) = permute(trialresults(4,:,:),[1 4 2 3]);
iISR(trial,ind_param,:) = trial_iISR;
end

end

%%
order = [1 15 12 10 11 13 14]; % 6 7 8 9 10 11 12 13];
legenda = {'ini', 'FastIVE', 'iFastIVE', 'PSIVE','iPSIVE', ...
    'CaponIVE', 'iCaponIVE'};
lines = {':','--','-','--','-','--','-','-','-.','-.','-.',':',':',':','-'};
marks = {'none','^','v','o','s','x','+','s','o','x','+','none','x','o','none'};
cols = {[0 0 0],[1 0.3 0],[1 0.3 0],[0 0.6 0],[0 0.6 0],[0 0 0],[0 0 0],[0 0.6 0],...
    [0.6 0 0.6],[0.6 0 0.6],[0.7 0.5 0],'k','k',[0.7 0.7 0.7],'k'};
widths = {3,2,2,2,2,2,2,2,2,2,2,3,3,3,3};

f = figure; f.Position(2:4) = [400 600 500];
subplot('Position',[0.15 0.15 0.33 0.65])
aux = 10*log10(oISR(:,:,:,order))<-3; 
h = semilogx(parameters,squeeze(mean(mean(aux,1,'omitnan'),3,'omitnan')*100));
set(gca,'FontSize',14)
for k = 1:length(h)
    h(k).LineStyle = lines{k};
    h(k).LineWidth = widths{k};
    h(k).Color = cols{k};
    h(k).Marker = marks{k};
    h(k).MarkerSize = 8;
end
grid on
ylabel('success rate [%]','FontSize',16)
xlabel('N','FontSize',16)

subplot('Position',[0.6 0.15 0.33 0.65])
aux = oISR(:,:,:,order);
aux(10*log10(aux)>-3) = nan; 
h = semilogx(parameters,-10*log10(squeeze(mean(mean(aux,1,'omitnan'),3,'omitnan'))));
set(gca,'FontSize',14)
for k = 1:length(h)
    h(k).LineStyle = lines{k};
    h(k).LineWidth = widths{k};
    h(k).Color = cols{k};
    h(k).Marker = marks{k};
    h(k).MarkerSize = 8;
end
grid on
ylabel('SIR [dB]','FontSize',16)
xlabel('N','FontSize',16)
legend(legenda,'Position',[0.13 0.83 0.75 0.1],'Orientation','horizontal','NumColumns',5);
