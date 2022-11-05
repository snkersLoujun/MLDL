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

function L = LLR1(X,Y, type,k)

[n, ~] = size(X);

if type == 1
    % 1. Neighborhood preserving regularizer with Boolean values
    %======= construct the affinity matrix of sample space ===========%
    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'Binary';
    options.t = 1;
    W = constructW(Y, options);
    L = diag(sum(W,2))-W;
    L=X'*L*X;
    clear W options
 
    
elseif type == 2
    
    % Neighborhood preserving regularizer with cosine
    %======= construct the affinity matrix of sample space ===========%
    options = [];
    options.Metric = 'Cosine';
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'Cosine';
    options.t = 1;
    W = constructW(Y, options);
    L = diag(sum(W,2))-W;
    L=X'*L*X;
    clear W options
    
    
    
elseif type == 3
    
    % Neighborhood preserving regularizer with Heat kernel
    %======= construct the affinity matrix of sample space ===========%
    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
%     options.k = 5;
    options.k = k;
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    W = constructW(Y, options);
    L = diag(sum(W,2))-W;
    L=X'*L*X;
    clear W options
    
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
     L=X'*L*X;
elseif type == 5
    W = reconWeight(Y', 5);
    W=X'*W*X;
    L = (eye(n)-W)'*(eye(n)-W);
     L=X'*L*X;
    clear W 
elseif type == 6
    W = constructW_PKN(X', 10, 0);
    W=(W'+W)/2;
    D=diag(sum(W,2));
    L=D-W;
    clear W 
elseif type == 7
    [n,~]=size(X);
    [Nk_index,Rk_index]=reverseNeighbor(X,3,n);
    [~,W]=calculateArrT(Nk_index,Rk_index,0.5,n);
    W=(W'+W)/2;
    D=diag(sum(W,2));
    L=D-W;
    clear W 
else
    fprintf('The local learning regularizer you choose does not exist!');
end

end