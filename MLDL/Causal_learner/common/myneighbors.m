function ns = myneighbors(adj_mat, i)
% NEIGHBORS Find the parents and children of a node in a graph.
% ns = neighbors(adj_mat, i)

%ns = myunion(children(adj_mat, i), parents(adj_mat, i));
ns = unique([find(adj_mat(i,:)) find(adj_mat(:,i))']);

