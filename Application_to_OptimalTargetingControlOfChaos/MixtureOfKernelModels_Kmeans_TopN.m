function [Models] = MixtureOfKernelModels_Kmeans_TopN( TrainData, Params, Model_Init)
%MixtureOfKernelModels_Kmeans
% Identical to EM, but only the most likely TopN models are considered for
% each point.
%   (If TopN = N_Models then the algorithm is identical to the full EM.)

X_train = TrainData.X;
U_train = TrainData.U;
Y_train = TrainData.Y;

N_Models = Params.N_Models;
TopN = Params.TopN;
lambda = Params.lambda;
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

init_yn = Model_Init.init_yn;

if strcmpi(init_yn,'y')
    
    MdlCounts = Model_Init.MdlCounts;
    MdlIndices = Model_Init.MdlIndices;
    Mu_x = Model_Init.Mu_x;
    Mu_y = Model_Init.Mu_y;
    M_matrices = Model_Init.M_matrices;
    K = Model_Init.K_mat;
    Sigma_Mdls = Model_Init.Sigma_Mdls;
    SigInv_Mdls = Model_Init.SigInv_Mdls;
    SigDet_Mdls = Model_Init.SigDet_Mdls;
    R_Mdls = Model_Init.R_Mdls;
    Rinv_Mdls = Model_Init.Rinv_Mdls;
    RDet_Mdls = Model_Init.RDet_Mdls;
    Phi_Mdls = Model_Init.phi_Mdls;
    
    % Initialize joint probabilities for calculating the expectation
    P_yxuz = zeros(N_train, N_Models);
    
    % Initialize distribution over models at each training point
    Q_bar = zeros(N_train, N_Models);
    
    for i=1:N_Models
        
        % indices belonging to cluster
        N_cluster = MdlCounts(i);
        indices = MdlIndices(i,1:N_cluster);
        
        % Model error at all points
        V_i = Y_train - ( Mu_y(:,i)*ones(1,N_train) + ...
            M_matrices(:,indices,i)*K(indices,:) );
        
        % Joint probabilities
        for j = 1:N_train
            P_yxuz(j,i) = Phi_Mdls(i) * NormalDist(V_i(:,j), ...
                zeros(x_dim,1), Rinv_Mdls(:,:,i), RDet_Mdls(i)) * ...
                NormalDist(X_train(:,j), Mu_x(:,i), SigInv_Mdls(:,:,i), ...
                SigDet_Mdls(i));
        end
    end
    
else
    
    fprintf('\n *** Evaluating kernel... \n')
    kernel = Params.kernel;
    K = zeros(N_train,N_train);
    for i = 1:N_train
        for j = i:N_train
            z1 = [X_train(:,i); U_train(:,i); 1];
            z2 = [X_train(:,j); U_train(:,j); 1];
            K(i,j) = kernel(z1, z2);
            K(j,i) = conj(K(i,j));
        end
    end
    
    % centroids in domain
    Mu_x = zeros(x_dim,N_Models);
    % centroids in range
    Mu_y = zeros(x_dim,N_Models);
    % Linear system matrices
    M_matrices = zeros(x_dim,N_train,N_Models);
    
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
    
    % Initialize joint probabilities for calculating the expectation
    P_yxuz = zeros(N_train, N_Models);
    
    % Initialize distribution over models at each training point
    Q_bar = zeros(N_train, N_Models);
    
    % allocate points to models initially based on distance
    for j = 1:N_train
        x = X_train(:,j);
        Dx = x*ones(1,N_Models) - Mu_x;
        dists = sqrt(sum(Dx.^2,1));
        [~,idx_mdl] = min(dists);
        
        P_yxuz(j,idx_mdl) = 1.0;
        Q_bar(j,idx_mdl) = 1.0;
        
        MdlCounts(idx_mdl) = MdlCounts(idx_mdl) + 1;
        MdlIndices(idx_mdl, MdlCounts(idx_mdl)) = j;
    end
    
    % initialize priors
    Phi_Mdls = MdlCounts/sum(MdlCounts);
    
end

%% Fit linear models using K-Means clustering

fprintf('\n *** Beginning K-Means convergence loop... \n')
Mdl_error = zeros(1,TopN);
iter = 0;
Delta_Centroids = 1e6*ones(x_dim,N_Models);
while any(sqrt(sum(Delta_Centroids.^2,1)) > Tol_centroid) && iter < Iter_Max
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Expectation step
    %
    MdlCounts = zeros(1,N_Models);
    MdlIndices = ones(N_Models,N_train);
    
    for j = 1:N_train
        % determine TopN best models for point j based on maximum
        % likelihood
        [~,mdl_idx_sort] = sort(P_yxuz(j,:), 'descend');
        P_yxuz(j,mdl_idx_sort(TopN+1:end)) = 0;
        
        % determine the normalized weights
        Q_bar(j,:) = P_yxuz(j,:)/sum(P_yxuz(j,:));
        
        % assign point j to TopN models
        for k = 1:TopN
            MdlCounts(mdl_idx_sort(k)) = MdlCounts(mdl_idx_sort(k)) + 1;
            MdlIndices(mdl_idx_sort(k), MdlCounts(mdl_idx_sort(k))) = j;
        end
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Locate centroids, fit Gaussians, and fit weighted linear models
    %
    for i = 1:N_Models
        % indices belonging to cluster
        N_cluster = MdlCounts(i);
        indices = MdlIndices(i,1:N_cluster);
        
        % weights on points in cluster
        w = Q_bar(indices,i); % weight of each point
        W = diag(w); % weight matrix
        
        % data points in cluster
        X_cluster = X_train(:, indices);
        U_cluster = U_train(:, indices);
        Y_cluster = Y_train(:, indices);
        
        % calculate centroids
        Mu_x_new = X_cluster*w / sum(w);
        Delta_Centroids(:,i) = Mu_x_new - Mu_x(:,i);
        Mu_x(:,i) = Mu_x_new;
        
        Mu_y(:,i) = Y_cluster*w / sum(w);
        
        %
        % fit nonlinear kernel model
        %
        DY = Y_cluster - Mu_y(:,i)*ones(1,N_cluster);
        K_sub = K(indices,indices); % kernel sub-matrix
        M_matrices(:,:,i) = zeros(x_dim,N_train);
        M_matrices(:,indices,i) = DY*W*pinv(K_sub*W + ...
            lambda*eye(N_cluster));
        
        % Find error covariance
        DY_error = DY - M_matrices(:,indices,i)*K_sub;
        R_i = (DY_error*W*DY_error')/sum(w);
        
        [U,S,V] = svd(R_i); % Correct degenerate Gaussians
        eps = eps_R*ones(x_dim,1);
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        R_Mdls(:,:,i) = U*S*V';
        Rinv_Mdls(:,:,i) = V*Sinv*U';
        RDet_Mdls(i) = prod(s);
        
        %
        % calculate MLE Gaussian covariances
        %
        Dx_pts = X_cluster(:,:) - Mu_x(:,i)*ones(1,N_cluster);
        Sig = (Dx_pts*W*Dx_pts')/sum(w);
        
        [U,S,V] = svd(Sig); % Correct degenerate Gaussians
        eps = eps_Sig*[1;1];
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        Sigma_Mdls(:,:,i) = U*S*V';
        SigInv_Mdls(:,:,i) = V*Sinv*U';
        SigDet_Mdls(i) = prod(s);
        
        %
        % Weighted Multinoulli
        %
        Phi_Mdls(i) = sum(w)/N_train;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update joint probabilities for calculating the expectation
        %
        
        % Model error at all points
        V_i = Y_train - ( Mu_y(:,i)*ones(1,N_train) + ...
            M_matrices(:,indices,i)*K(indices,:) );
        
        % Joint probabilities
        for j = 1:N_train
            P_yxuz(j,i) = Phi_Mdls(i) * NormalDist(V_i(:,j), ...
                zeros(x_dim,1), Rinv_Mdls(:,:,i), RDet_Mdls(i)) * ...
                NormalDist(X_train(:,j), Mu_x(:,i), SigInv_Mdls(:,:,i), ...
                SigDet_Mdls(i));
        end
    end
    
    iter = iter + 1;
    fprintf('Iteration %d : Max Centroid Change = %.3e \n', ...
        iter, max(sqrt(sum(Delta_Centroids.^2,1))))
    
end


%%
Models.MdlCounts = MdlCounts;
Models.MdlIndices = MdlIndices;
Models.Mu_x = Mu_x;
Models.Mu_y = Mu_y;
Models.M_matrices = M_matrices;
Models.K_mat = K;
Models.Sigma_Mdls = Sigma_Mdls;
Models.SigInv_Mdls = SigInv_Mdls;
Models.SigDet_Mdls = SigDet_Mdls;
Models.R_Mdls = R_Mdls;
Models.Rinv_Mdls = Rinv_Mdls;
Models.RDet_Mdls = RDet_Mdls;
Models.phi_Mdls = Phi_Mdls;

end
