function [J,I,theta,value] = MLFS_LLL21(x, y, k,lambda,beta,beta1)
% [m, ~] = size(x);
% x = [ones(m, 1), x];
y(y==-1)=0;
[m, n] = size(x);
[~, n1] = size(y);
g = inline('1.0 ./ (1.0 + exp(-z))');
[~,L] = cLLR(x',3);
% L1 = LLR(y,type);

% d=pdist2(x,x);
% L1=d.*L1;
L1 = LLR1(x,y,3,5);
theta_n =zeros(n, n1);
theta=theta_n;
J = [];
for i=1:300
    d=sum(theta_n.^2,2);
    idx=find(d==0);
    if idx>0
        D=diag(ones(1,n));
    else
        D=diag(1./sqrt(sum(theta_n.^2,2)));
    end
    z = x * theta;
    h = g(z);
    J(i) =(1/(m*n1))*(sum(sum(-y.*log(h) - (1-y).*log(1-h)))+0.5*beta1*trace(theta'*L1*theta)+lambda*sum(sqrt(sum(theta.^2,2)))+0.5*beta*trace(theta'*L*theta));%ËðÊ§º¯Êý
    grad = (1/(m*n1))*(x' * (h-y)+lambda.*D*theta+beta.*(L*theta)+beta1.*(L1*theta));
    H = (1/(m*n1))*(x' *(diag(sum(h.*(1-h),2)))* x+lambda.*D+beta.*L+beta1.*L1);
    dx=-pinv(H)*grad;
    theta_n = theta + dx;
    theta=theta_n;
end
S=theta(2:end,:);
tempVector = sum(S.*S,2);
% tempVector = mean(abs(S),2);
[~, value] = sort(tempVector, 'descend'); % sort W in a descend order
clear tempVector;
I = value(1:k);
end