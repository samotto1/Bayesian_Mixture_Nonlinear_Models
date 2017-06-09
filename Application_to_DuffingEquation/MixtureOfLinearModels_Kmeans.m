function [LinModels] = MixtureOfLinearModels_Kmeans( TrainData, Params)
%MixtureOfLinearModels_Kmeans

X_train = TrainData.X;
U_train = TrainData.U;
Y_train = TrainData.Y;

N_Models = Params.N_Models;
N_nearest = Params.N_Nearest;
Iter_Max = Params.Iter_Max;
Tol_centroid = Params.Tol_centroid;
eps_Sig = Params.eps_Sig;
eps_R = Params.eps_R;

N_train = size(X_train,2);
x_dim = size(X_train,1);
u_dim = size(U_train,1);

NormalDist = @(x, mu, SigInv, SigDet) ...
    1/( (2*pi)^(length(x)/2)*sqrt(SigDet) ) *...
    exp( -0.5*(x-mu)'*SigInv*(x-mu) );


%% Initialization
% centroids in domain
Mu_x = zeros(x_dim,N_Models);
% centroids in range
Mu_y = zeros(x_dim,N_Models);
% Linear system matrices
A_matrices = zeros(x_dim,x_dim,N_Models);
% Control sensitivity matrices
B_matrices = zeros(x_dim,u_dim,N_Models);

% Covariance of MLE Gaussians
Sigma_Mdls = zeros(x_dim,x_dim,N_Models);
SigInv_Mdls = zeros(x_dim,x_dim,N_Models);
SigDet_Mdls = zeros(1,N_Models);

% Error covariance of linear models
R_Mdls = zeros(x_dim,x_dim,N_Models);
Rinv_Mdls = zeros(x_dim,x_dim,N_Models);
RDet_Mdls = zeros(1,N_Models);

% Prior likelihoods of models
phi_Mdls = ones(1,N_Models)/N_Models;

% Initialize point allocations
MdlCounts = zeros(1,N_Models);
MdlIndices = ones(N_Models,N_train);

% pick initial centroids randomly
L = floor(N_train/N_Models);
for i = 1:N_Models
    ii = L*(i-1) + floor(L*rand(1)) + 1;
    Mu_x(:,i) = X_train(:,ii);
end

% allocate points to models initially based on distance
for j = 1:N_train
    x = X_train(:,j);
    Dx = x*ones(1,N_Models) - Mu_x;
    dists = sqrt(sum(Dx.^2,1));
    [~,idx_mdl] = min(dists);
    
    MdlCounts(idx_mdl) = MdlCounts(idx_mdl) + 1;
    MdlIndices(idx_mdl, MdlCounts(idx_mdl)) = j;
end

% initialize priors
Phi_Mdls = MdlCounts/sum(MdlCounts);

%% Fit linear models using K-Means clustering

fprintf('\n *** Beginning K-Means convergence loop... \n')
Mdl_error = zeros(1,N_nearest);
iter = 0;
Delta_Centroids = 1e6*ones(x_dim,N_Models);
while any(sqrt(sum(Delta_Centroids.^2,1)) > Tol_centroid) && iter < Iter_Max

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Locate centroids, fit Gaussians, and fit weighted linear models
    %
    for i = 1:N_Models
        % indices belonging to cluster
        indices = MdlIndices(i,1:MdlCounts(i));
        
        % data points in cluster
        X_cluster = X_train(:, indices);
        U_cluster = U_train(:, indices);
        Y_cluster = Y_train(:, indices);
        
        % calculate centroids
        Mu_x_new = mean(X_cluster(:,:),2);
        Delta_Centroids(:,i) = Mu_x_new - Mu_x(:,i);
        Mu_x(:,i) = Mu_x_new;
        Mu_y(:,i) = mean(Y_cluster(:,:),2);
        
        % calculate MLE Gaussian covariances
        Dx_pts = X_cluster(:,:) - Mu_x(:,i)*ones(1,MdlCounts(i));
        Sig = 1.0/MdlCounts(i) * (Dx_pts*Dx_pts');
        
        [U,S,V] = svd(Sig); % Correct degenerate Gaussians
        eps = eps_Sig*[1;1];
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        Sigma_Mdls(:,:,i) = U*S*V';
        SigInv_Mdls(:,:,i) = V*Sinv*U';
        SigDet_Mdls(i) = prod(s);
        
        % calculate weights using Gaussians
        w = zeros(1,MdlCounts(i)); % weight of each point
        for ii = 1:MdlCounts(i)
            Dx = Dx_pts(:,ii); % dist from centroid
            w(ii) = NormalDist(Dx, zeros(x_dim,1), SigInv_Mdls(:,:,i),...
                SigDet_Mdls(i));
        end
         
        % fit linear model
        DX = [X_cluster - Mu_x(:,i)*ones(1,MdlCounts(i));
            U_cluster];
        DY = Y_cluster - Mu_y(:,i)*ones(1,MdlCounts(i));
        
        DXW = DX .* (ones(x_dim+u_dim,1)*sqrt(w));
        DYW = DY .* (ones(x_dim,1)*sqrt(w));
        
        M = DYW*pinv(DXW);
        A_matrices(:,:,i) = M(:,1:x_dim);
        B_matrices(:,:,i) = M(:,x_dim+1:end);
        
        
        % Find error covariance
        DY_error = DY - M*DX;
        R_i = 1/MdlCounts(i) * (DY_error*DY_error');
        
        [U,S,V] = svd(R_i); % Correct degenerate Gaussians
        eps = eps_R*ones(x_dim,1);
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        R_Mdls(:,:,i) = U*S*V';
        Rinv_Mdls(:,:,i) = V*Sinv*U';
        RDet_Mdls(i) = prod(s);
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Assign points to models based on minimum error
    %
    
    % First empty the previous point assignments
    MdlCounts = zeros(1,N_Models);
    MdlIndices = ones(N_Models,N_train);
    
    % Assign points to models
    for j = 1:N_train
        x = X_train(:,j);
        u = U_train(:,j);
        y = Y_train(:,j);
        
        % Locate the N_nearest nearest models
        Dx = x*ones(1,N_Models) - Mu_x;
        Dist = sqrt(sum(Dx.^2,1));
        [~,inds] = sort(Dist);
        MdlInds = inds(1:N_nearest);
        
        % Evaluate errors using each model 
        for ii = 1:N_nearest
            idx_mdl = MdlInds(ii);
            
            y_hat = A_matrices(:,:,idx_mdl)*(x - Mu_x(:,idx_mdl)) + ...
                B_matrices(:,:,idx_mdl)*u + Mu_y(:,idx_mdl);
            y_error = y - y_hat;
            
            Mdl_error(ii) = sqrt(y_error'*y_error);
        end
        
        % Assign to model based on minimum error
        [~, ii_star] = min(Mdl_error);
        idx_mdl_star = MdlInds(ii_star);
        MdlCounts(idx_mdl_star) = MdlCounts(idx_mdl_star) + 1;
        MdlIndices(idx_mdl_star, MdlCounts(idx_mdl_star)) = j;
    end
    
    % update prior proabilities of each model
    Phi_Mdls = MdlCounts/sum(MdlCounts);
    
    iter = iter + 1;
    fprintf('Iteration %d : Max Centroid Change = %.3e \n', ...
        iter, max(sqrt(sum(Delta_Centroids.^2,1))))
    
end


%%
LinModels.MdlCounts = MdlCounts;
LinModels.MdlIndices = MdlIndices;
LinModels.Mu_x = Mu_x;
LinModels.Mu_y = Mu_y;
LinModels.A_matrices = A_matrices;
LinModels.B_matrices = B_matrices;
LinModels.Sigma_Mdls = Sigma_Mdls;
LinModels.SigInv_Mdls = SigInv_Mdls;
LinModels.SigDet_Mdls = SigDet_Mdls;
LinModels.R_Mdls = R_Mdls;
LinModels.Rinv_Mdls = Rinv_Mdls;
LinModels.RDet_Mdls = RDet_Mdls;
LinModels.phi_Mdls = phi_Mdls;

end
