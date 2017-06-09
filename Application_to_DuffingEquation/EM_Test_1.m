clear
clc
close all

N_pts = 1000;
%X = -ones(1,N_pts) + 2*rand(1,N_pts);
X = [-2.5 + 1*randn(1,N_pts/2), 2.5 + 1*randn(1,N_pts/2)];
U = zeros(1,N_pts);
Y_raw = sin(pi*X);
Y = Y_raw + 0.2*randn(1,N_pts);

N_LinModels = 15;
Iter_Max = 500;
Tol_centroid = 1e-4;
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

Params.N_LinModels = N_LinModels;
Params.Iter_Max = Iter_Max;
Params.Tol_centroid = Tol_centroid;
Params.eps_Sig = eps_Sig;
Params.eps_R = eps_R;

Params.sig_init = sig_init;

[LinModels] = MixtureOfLinearModels_EM( TrainData, Params);

Mu_x = LinModels.Mu_x;
Mu_y = LinModels.Mu_y;
A_matrices = LinModels.A_matrices;
B_matrices = LinModels.B_matrices;
Sigma_Mdls = LinModels.Sigma_Mdls;
SigInv_Mdls = LinModels.SigInv_Mdls;
SigDet_Mdls = LinModels.SigDet_Mdls;
R_Mdls = LinModels.R_Mdls;
Rinv_Mdls = LinModels.Rinv_Mdls;
RDet_Mdls = LinModels.RDet_Mdls;
phi_Mdls = LinModels.phi_Mdls;


% test the model

% Initialize point allocations
MdlCounts = zeros(1,N_LinModels);
MdlIndices = ones(N_LinModels,N_pts);

% Assign points to models and estimate using maximum likelihood model
Y_hat1 = zeros(1,N_pts);
Y_hat2 = zeros(1,N_pts);
P_Mdls = zeros(1,N_LinModels);
for j = 1:N_pts
    x = X(j);
    u = U(j);
    
    % Evaluate likelihood each model
    for i = 1:N_LinModels
        
        P_Mdls(i) = phi_Mdls(i) * NormalDist(x, Mu_x(:,i), ...
            SigInv_Mdls(:,:,i), SigDet_Mdls(i));
        
    end
    P_Mdls = P_Mdls/sum(P_Mdls);
    % Assign to model based on max likelihood
    [~, i_star] = max(P_Mdls);
    MdlCounts(i_star) = MdlCounts(i_star) + 1;
    MdlIndices(i_star, MdlCounts(i_star)) = j;
    
    Y_hat1(j) = A_matrices(:,:,i_star)*(x - Mu_x(:,i_star)) + ...
    	B_matrices(:,:,i_star)*u + Mu_y(:,i_star);
    
    for i = 1:N_LinModels
        
        Y_hat2(j) =Y_hat2(j) + P_Mdls(i)*( A_matrices(:,:,i)*(x - Mu_x(:,i)) + ...
    	B_matrices(:,:,i)*u + Mu_y(:,i) );
        
    end
end

% Plot the different points
figure()
C = lines(N_LinModels);
for i = 1:N_LinModels
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
C = lines(N_LinModels);
for i = 1:N_LinModels
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
