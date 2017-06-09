clear
clc
close all

kernel = @(x,z) (x'*z + 1).^4;
%kernel = @(x,z) exp(-(x-z)'*(x-z)/4^2);

N_pts = 500;
%X = -ones(1,N_pts) + 2*rand(1,N_pts);
X = [-2.5 + 0.5*randn(1,N_pts/2),...
    2.5 + 0.5*randn(1,N_pts/2)];
%X = -ones(1,N_pts) + 1*randn(1,N_pts);
U = zeros(1,N_pts);
Y_raw = 0.0*X.^2 + sin(1*pi*X);
Y = Y_raw + 0.05*randn(1,N_pts);

N_Models = 8;
N_SubTrain = 200; % number of points to use in Kernel regression
lambda = 0.001; % Ridge regression regularization parameter
Iter_Max = 500;
Tol_centroid = 1e-5;
eps_Sig = 1e-6;
eps_R = 1e-6;
sig_init = 0.1;

NormalDist = @(x, mu, SigInv, SigDet) ...
    1/( (2*pi)^(length(x)/2)*sqrt(SigDet) ) *...
    exp( -0.5*(x-mu)'*SigInv*(x-mu) );

% fit the linear models
TrainData.X = X;
TrainData.U = U;
TrainData.Y = Y;

Params.N_LinModels = N_Models;
Params.N_SubTrain = N_SubTrain;
Params.lambda = lambda;
Params.Iter_Max = Iter_Max;
Params.Tol_centroid = Tol_centroid;
Params.eps_Sig = eps_Sig;
Params.eps_R = eps_R;

Params.sig_init = sig_init;

Params.kernel = kernel;

[LinModels] = MixtureOfKernelModels_EM( TrainData, Params);

Mu_x = LinModels.Mu_x;
Mu_y = LinModels.Mu_y;
M_matrices = LinModels.M_matrices;
Sigma_Mdls = LinModels.Sigma_Mdls;
SigInv_Mdls = LinModels.SigInv_Mdls;
SigDet_Mdls = LinModels.SigDet_Mdls;
R_Mdls = LinModels.R_Mdls;
Rinv_Mdls = LinModels.Rinv_Mdls;
RDet_Mdls = LinModels.RDet_Mdls;
phi_Mdls = LinModels.phi_Mdls;


% test the model
K = zeros(N_pts, N_pts);
for i = 1:N_pts
    for j = 1:N_pts
        K(i,j) = kernel([X(:,i); U(:,i);1], [X(:,j); U(:,j);1]);
    end
end

% Initialize point allocations
MdlCounts = zeros(1,N_Models);
MdlIndices = ones(N_Models,N_pts);

% Assign points to models and estimate using maximum likelihood model
Y_hat1 = zeros(1,N_pts);
Y_hat2 = zeros(1,N_pts);
P_Mdls = zeros(1,N_Models);
for j = 1:N_pts
    x = X(j);
    u = U(j);
    
    % Evaluate likelihood each model
    for i = 1:N_Models
        
        P_Mdls(i) = phi_Mdls(i) * NormalDist(x, Mu_x(:,i), ...
            SigInv_Mdls(:,:,i), SigDet_Mdls(i));
        
    end
    P_Mdls = P_Mdls/sum(P_Mdls);
    % Assign to model based on max likelihood
    [~, i_star] = max(P_Mdls);
    MdlCounts(i_star) = MdlCounts(i_star) + 1;
    MdlIndices(i_star, MdlCounts(i_star)) = j;
    
    Y_hat1(j) = M_matrices(:,:,i_star)*K(:,j) + Mu_y(:,i_star);
    
    for i = 1:N_Models      
        Y_hat2(j) =Y_hat2(j) + P_Mdls(i)*...
            ( M_matrices(:,:,i)*K(:,j) + Mu_y(:,i) );     
    end
end

% Plot the different points
figure()
C = lines(N_Models);
for i = 1:N_Models
    indices = MdlIndices(i,1:MdlCounts(i));
    [~,iinds] = sort(X(indices));
    hold on
    plot(X(indices(iinds)), Y(indices(iinds)), '.', ...
        'Color', C(i,:), 'LineWidth',1.5)
    plot(Mu_x(1,i), Mu_y(1,i), 'o', ...
        'Color', C(i,:), 'LineWidth',2)
    plot(X(indices(iinds)), Y_hat1(indices(iinds)), '-', ...
        'Color', C(i,:), 'LineWidth',1.5)
end
[~,sort_inds] = sort(X);
%plot(X(sort_inds), Y_raw(sort_inds), 'k-', 'LineWidth',1.5)
hold off
title('Final EM Clusters and Centroids')
grid on
drawnow


figure()
C = lines(N_Models);
for i = 1:N_Models
    indices = MdlIndices(i,1:MdlCounts(i));
    [~,iinds] = sort(X(indices));
    hold on
    plot(X(indices(iinds)), Y(indices(iinds)), '.', ...
        'Color', C(i,:), 'LineWidth',1.5)
    plot(Mu_x(1,i), Mu_y(1,i), 'o', ...
        'Color', C(i,:), 'LineWidth',2)
    plot(X(indices(iinds)), Y_hat2(indices(iinds)), '-', ...
        'Color', C(i,:), 'LineWidth',1.5)
end
%plot(X(sort_inds), Y_raw(sort_inds), 'k-', 'LineWidth',1.5)
hold off
title('Final EM Clusters and Centroids')
grid on
drawnow
