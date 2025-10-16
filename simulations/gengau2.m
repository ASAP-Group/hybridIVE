function x=gengau2(alfa,m,n);
%
if isinf(alfa)
    x=rand(m,n);
else
r=0.5*rand(1,m*n)+0.5;
beta=sqrt(gamma(3/alfa)/gamma(1/alfa));
y=r/beta;
ymin=1e-20*ones(1,m*n); ymax=1000*ones(1,m*n);
for iter=1:33
    cdf=0.5+0.5*gammainc((beta*y).^alfa,1/alfa);
    indplus=find(cdf>r);
    if any(indplus)
       ymax(indplus)=y(indplus);
    end
    indminus=find(cdf<r);
    if any(indminus)
       ymin(indminus)=y(indminus); 
    end   
    y=0.5*(ymax+ymin);
end    
ind=rand(1,m*n)>0.5;
if any(ind)
   y(ind)=-y(ind);
end   
x=reshape(y,m,n);
end
