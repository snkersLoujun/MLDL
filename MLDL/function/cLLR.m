%% Compute local learning regularizers (LLR)
%
% Written by Shiping Wang (shipingwangphd@gmai.com), December 14, 2014.
% 
%%%%%%%%%%%% Input %%%%%%%%%%%%%%%%%%%%%
% X: data matrix (n x d)
% type: choose the type of local learning regularizer 
%        1: 0-1 graph Laplacian L = D - W
%        2: Cosine graph Laplacian L = D - W
%        3. Kernel graph Laplacian (Heat kernel) L = D - W
%        4. Sparsity preserving regularizer 
%        5. [0,1] neighborhood preserving regularizer L = (I-W)'(I-W)
%        
%%%%%%%%%%%% Output %%%%%%%%%%%%%%%%%%%%%
% L: local learning regularizer for samples (n x n)

function [L] = cLLR(X, type)

[n, ~] = size(X);%只输出X的行数

if type == 1
    % 1. Neighborhood preserving regularizer with Boolean values布尔值
    %======= construct the affinity matrix of sample space ===========%
    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'Binary';
    options.t = 1;
    W = constructW(X, options);
    B=diag(sum(W,2));
    L = B-W;
%     clear W options
 
    
elseif type == 2
    
    % Neighborhood preserving regularizer with cosine
    %======= construct the affinity matrix of sample space ===========%
    options = [];
    options.Metric = 'Cosine';
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'Cosine';
    options.t = 1;
    W = constructW(X, options);
    L = diag(sum(W,2))-W;
    clear W options
    
    
    
elseif type == 3
    
    % Neighborhood preserving regularizer with Heat kernel
    %======= construct the affinity matrix of sample space ===========%
    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    W = constructW(X, options);
%     B=diag(sum(W,2));
%     L = B-W;
    L = diag(sum(W,2))-W;
    clear W options
    %%
elseif type == 4
    
    % Set the paramter alpha
    alpha = 0.1;
    
    % compute the coefficient matrix Y \in R^{n-1,n} 
    Y = zeros(n-1,n);
    for i = 1:n
    Z = X';
    Z(:,i) = [];
    Y(:,i) = (Z'*Z+alpha*eye(n-1))\(Z'*X(i,:)');
    clear Z
    end

    % get the weighted matrix W1 \in R^{n,n}
    A = reshape(Y,1,n*(n-1)); % reshape the matrix
    [I, J] = find((ones(n)-eye(n))>0); % compute the index set
    W = full(sparse(I,J,A,n,n));
    clear A I J Y Z alpha
    L = (eye(n)-W)'*(eye(n)-W);
    %%
elseif type == 5
    %%
    W = reconWeight(X', 5);
    L = (eye(n)-W)'*(eye(n)-W);
    clear W 
   
else
    fprintf('The local learning regularizer you choose does not exist!');
end

end
    



