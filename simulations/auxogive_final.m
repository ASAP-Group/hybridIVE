function [shat_out,out] = auxogive_final(x,opt)

% if exist('pagemtimes','builtin')
%     mtimesx = @newmtimesx;
% end

epsilon = 0.000001;

[d, N, M] = size(x); 

if ~isfield(opt,'MaxIt')
    opt.MaxIt = 50; %pokud neni urcen pocet iteraci; default 50
end
if ~isfield(opt,'BlockSize')
    opt.BlockSize = 200; %delka blocku default 200
end

if ~isfield(opt,'wini')
    opt.wini = randn(d,M)+1j*randn(d,M); %pokud neni inicializace, inicializace je nahodna
end

b = opt.BlockSize; 
Nblocks = floor(N/b);
blocks = [(0:Nblocks)*b,N]; %rozdeleni framu do bloku

%% Var preparation

if isfield(opt,'c_s') %pokud je pridan cisty source
    out.c_s = opt.c_s(1,:,:);
    c_s_block = zeros(d,b,Nblocks,M);
end
if isfield(opt,'c_n')%pokud je pridan cisty noise
    out.c_n = opt.c_n(1,:,:);
    c_n_block = zeros(d,b,Nblocks,M);
end

if isfield(opt,'P') %pokud se pilotuje
    PP = zeros(M,b,Nblocks);
end

if isfield(opt,'tau') && isfield(opt,'P') %pokud je pozadovan wiener
    if length(opt.tau) < 2 % pokud je jina vaha pro pilotovane framy
        opt.tau(2)
    end
end

%inicializace promenych
shat = x(1,:,:);
shat_out =  shat;
a = zeros(d,M,Nblocks);
xx = zeros(d,b,Nblocks,M);

NumIt = 0;

%% Separation
%Rozblockovani signalu
for block = 1:Nblocks
    xx(:,:,block,:) =  x(:,blocks(block)+1:blocks(block+1),:);
    if isfield(opt,'P')
        PP(:,:,block) = opt.P(:,blocks(block)+1:blocks(block+1));
    end
    if isfield(opt,'c_s')
        c_s_block(:,:,block,:) = opt.c_s(:,blocks(block)+1:blocks(block+1),:);
    end
    if isfield(opt,'c_n')
        c_n_block(:,:,block,:) = opt.c_n(:,blocks(block)+1:blocks(block+1),:);
    end
    
end

Cx = mtimesx(xx,xx,'C')/b;
w= (opt.wini)./(opt.wini(1,:,:,:));
w = permute(w,[1,3,4,2]);
crit = 0;

while crit < 1-epsilon && NumIt < opt.MaxIt
    NumIt = NumIt + 1;
    wold = w; 
    Cxw = mtimesx(Cx,w);
    lambdaw = mtimesx(w,'C',Cxw);
    a = mtimesx(Cxw,(1/lambdaw));
    shat = mtimesx(w,'C',xx);
    shat(abs(shat)<eps)=eps;
    
    if isfield(opt,'P')%preskalovani pilota pokud vstupuje do nelinearity
        l = mean(mean(sum(abs(shat).^2,4)));
        r = 1./sqrt(sum(abs(shat).^2,4)+l*PP);
        r = permute(r,[4 2 3 1]);
    else
        r = 1./sqrt(sum(abs(shat).^2,4));
    end
    
    v = mtimesx((xx).*r,(xx),'C')/b;
    up  = mtimesx(mtimesx(w,'C',v),w);
    a_scale = mtimesx(up,1/lambdaw);
    a_scaled = sum(mtimesx(a_scale,a),3);
    v_scaled = sum(mtimesx(v,1/lambdaw),3);
    for k = 1:M
        w(:,:,1,k) = (v_scaled(:,:,1,k).'\conj(a_scaled(:,:,1,k)))';%vypocet w
    end
    w = w./sqrt(sum(mtimesx(mean(v,3),w).*conj(w),1));%vazeni w
    crit = min(abs(sum(w.*conj(wold),1))./sqrt(sum(w.*conj(w),1).*sum(wold.*conj(wold),1)),[],4);
end
%a = mtimesx(Cxw,(1/lambdaw));
Cxw = mtimesx(Cx,w);
    lambdaw = mtimesx(w,'C',Cxw);
    a = mtimesx(Cxw,(1/lambdaw));
w = mtimesx(a(1,:,:,:),'G',w);% MDP
shat = mtimesx(w,'C',xx);

if isfield(opt,'c_s')
    c_s_block = mtimesx(w,'C',c_s_block);
end
if isfield(opt,'c_n')
    c_n_block = mtimesx(w,'C',c_n_block);
end

if isfield(opt,'tau')%wiener
    n_est = (xx(1,:,:,:)-shat);
    s_est = shat;
    tau = ones(1,b,Nblocks)*opt.tau(1);
    if isfield(opt,'P')
        tau(PP== 0) = opt.tau(2);
    end
    tau = repmat(tau,1,1,1,M);
    w_opt = max(abs(s_est).^2 - tau.*abs(n_est).^2,0.0001)./(abs(s_est).^2 + 0.0001);
    shat = w_opt.*s_est;
    out.w_wiener = w_opt;
    if isfield(opt,'c_s')
        c_s_block = w_opt.*c_s_block;
    end
    if isfield(opt,'c_n')
        c_n_block =  w_opt.*c_n_block;
    end
end
out.w = w;
out.a = a;
out.NumIt = NumIt;
%Rozblockovani zpet
for block = 1:Nblocks
    shat_out(:,blocks(block)+1:blocks(block+1),:) = shat(:,:,block,:);
    if isfield(opt,'c_s')
        out.c_s(:,blocks(block)+1:blocks(block+1),:) = c_s_block(:,:,block,:);
    end
    if isfield(opt,'c_n')
        out.c_n(:,blocks(block)+1:blocks(block+1),:) = c_n_block(:,:,block,:);
    end
end
%posledni neuplny block je separovan w z posledniho uplneho bloku
shat_out(:,blocks(end-1)+1:blocks(end),:) = mtimesx(w(:,:,end,:),'C',permute(x(:,blocks(end-1)+1:blocks(end),:),[1 2 4 3]));
if isfield(opt,'c_s')
    out.c_s(:,blocks(end-1)+1:blocks(end),:) = mtimesx(w(:,:,end,:),'C',permute(opt.c_s(:,blocks(end-1)+1:blocks(end),:),[1 2 4 3]));
end
if isfield(opt,'c_n')
    out.c_n(:,blocks(end-1)+1:blocks(end),:) = mtimesx(w(:,:,end,:),'C',permute(opt.c_n(:,blocks(end-1)+1:blocks(end),:),[1 2 4 3]));
end
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
