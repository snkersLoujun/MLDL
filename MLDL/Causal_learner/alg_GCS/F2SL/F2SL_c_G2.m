

function [G,time,test] = F2SL_c_G2(Data,alpha,ns,p,maxK)
%
% F2SL_c_G2 learns a causal graph on discrete data
%
% INPUT :
%       Data is the data matrix
%       alpha is the significance level
%       ns is the size array of each node
%       p is the number of nodes
%       maxK is the maximum size of conditioning set
%
% OUTPUT:
%       G is the causal graph
%       time is the runtime of the algorithm
%       test is the number of conditional independence tests
%
%


if (nargin == 2)
   ns=max(Data);
   [~,p]=size(Data);
   maxK=3;
end

start=tic;

test=0;

all_PC=cell(1,p);
all_sepset=cell(1,p);


% step 1�� establish seketon

PP=zeros(p,p);
for i=1:p
    PC=FCBF_PC_G2(Data,i,alpha,ns,p,maxK);
    PP(i,PC)=1;
end


% OR version
% PP_OR = sign(PP+PP');

% AND version
PP_AND = PP;
PP_AND((PP~=PP'))=0;

skeleton=PP_AND;

% step 2�� orient V-structures


for i=1:p
    all_PC{i}=find(skeleton(i,:)==1);
end

DAG=skeleton;
pdag=skeleton;
G=skeleton;


for i=1:p
    A=i;
    PCA=all_PC{A};
    
    for j=1:length(PCA)
        B=PCA(j);
        PCB=all_PC{B};
        
        for k=1:length(PCB)
            C=PCB(k);
            
            if ismember(C,all_PC{A})||C==A
                continue;
            end
            
            CanPC=all_PC{A};
            cutSetSize = 0;
            
            break_flag=0;
            
            while length(CanPC) >= cutSetSize &&cutSetSize<=maxK
                
                SS = subsets1(CanPC, cutSetSize);    
                for si=1:length(SS)
                    Z = SS{si};
                    
                    test=test+1;
                    pval=my_g2_test(C,A,Z,Data,ns,alpha);
                    
                    if pval>alpha
                        
                        all_sepset{A}{C}=Z;
                        
                        if ~ismember(B,Z)
                            test=test+1;
                            pval=my_g2_test(C,A,myunion(Z,B),Data,ns,alpha);
                            
                            if isnan(pval)||pval<=alpha
%                                 fprintf('V-structure: %d->%d<-%d\n',A,B,C);
                                
                                pdag(A,B) = -1; pdag(C,B) = -1; pdag(B,A) = 0; pdag(B,C) = 0;
                                G(A,B) = 1;     G(C,B) = 1;     G(B,A) = 0;    G(B,C) = 0;
                                
                                
                                
                            end
                        end
                        
                        break_flag=1;
                        break;
                    end
                end
                if break_flag
                    break;
                end
                cutSetSize = cutSetSize + 1;
            end
            
        end
    end
end


% step 3�� Meek

[DAG,pdag,G]=meeks(DAG,pdag,G,p);


G=cpdag_to_dag(G);

% draw_graph(G);

time=toc(start);
