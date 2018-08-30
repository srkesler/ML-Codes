% Shelli Kesler

% Agglomerative clusterer 

% https://www.mathworks.com/help/stats/hierarchical-clustering.html
% https://stackoverflow.com/questions/8016313/agglomerative-clustering-in-matlab
% http://www.sthda.com/english/articles/29-cluster-validation-essentials/96-determining-the-optimal-number-of-clusters-3-must-know-methods/#elbow-method

% import data
X = readtable('data.csv');


% scale data to SD = 1
% Z = abs(zscore(X));

%set seed
rng(1);

% hierarchical clustering
Q1 = pdist(X,'euclid'); % use Z for scaled data
Q2 = squareform(Q1);
linkmethod = 'complete'; % options: average, single, complete
Q3 = linkage(Q2,linkmethod);
dendrogram(Q3,'Labels',Country,'ColorThreshold','default')
xtickangle(45)

% Using k-means, cluster the data
% https://www.mathworks.com/help/stats/kmeans.html

%determine number of clusters with elbow plot
n=5; 
wss = zeros(1, n);
wss(1) = (size(X, 1)-1) * sum(var(X, [], 1));
for i=2:n
    T = clusterdata(X,'maxclust',i);
    wss(i) = sum((grpstats(T, T, 'numel')-1) .* sum(grpstats(X, T, 'var'), 2));
end
hold on
plot(wss)
plot(wss, '.')
xlabel('Number of clusters')
ylabel('Within-cluster sum-of-squares')


K = 4; % try different K's
repeat = 50; % number of times to repeat

opts = statset('Display', 'iter');
[idx,C] = kmeans(X, K, 'options',opts, ...
    'distance','sqEuclidean', 'EmptyAction','singleton', 'replicates',repeat);

% sumd = Within-cluster sums of point-to-centroid distances, returned as a 
%numeric column vector. sumd is a k-by-1 vector, where element j is the sum 
% of point-to-centroid distances within cluster j.


figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
hold on 
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
hold on
plot(X(idx==4,1),X(idx==4,2),'m.','MarkerSize',12)
hold on
plot(X(idx==5,1),X(idx==5,2),'y.','MarkerSize',12)
hold on
plot(X(idx==6,1),X(idx==6,2),'c.','MarkerSize',12)

plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
 

 legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

