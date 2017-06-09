function [LinModels] = MixtureOfLinearModels_EM( TrainData, Params)
%MixtureOfLinearModels_EM

X_train = TrainData.X;
U_train = TrainData.U;
Y_train = TrainData.Y;

N_LinModels = Params.N_LinModels;
Iter_Max = Params.Iter_Max;
Tol_centroid = Params.Tol_centroid;
eps_Sig = Params.eps_Sig;
eps_R = Params.eps_R;

sig_init = Params.sig_init;

N_train = size(X_train,2);
x_dim = size(X_train,1);
u_dim = size(U_train,1);

NormalDist = @(x, mu, SigInv, SigDet) ...
    1/( (2*pi)^(length(x)/2)*sqrt(SigDet) ) *...
    exp( -0.5*(x-mu)'*SigInv*(x-mu) );

%% Initialization
% centroids in domain
Mu_x = zeros(x_dim,N_LinModels);
% centroids in range
Mu_y = zeros(x_dim,N_LinModels);
% Linear system matrices
A_matrices = zeros(x_dim,x_dim,N_LinModels);
% Sensitivity matrices
B_matrices = zeros(x_dim,u_dim,N_LinModels);

% Covariance of MLE Gaussians
Sigma_Mdls = zeros(x_dim,x_dim,N_LinModels);
SigInv_Mdls = zeros(x_dim,x_dim,N_LinModels);
SigDet_Mdls = zeros(1,N_LinModels);

for i = 1:N_LinModels
    Sigma_Mdls(:,:,i) = sig_init*eye(x_dim);
    SigInv_Mdls(:,:,i) = (1.0/sig_init)*eye(x_dim);
    SigDet_Mdls(i) = sig_init^x_dim;
end

% Error covariance of linear models
R_Mdls = zeros(x_dim,x_dim,N_LinModels);
Rinv_Mdls = zeros(x_dim,x_dim,N_LinModels);
RDet_Mdls = zeros(1,N_LinModels);

% Prior likelihoods of models
phi_Mdls = ones(1,N_LinModels)/N_LinModels;

% pick initial centroids randomly
L = floor(N_train/N_LinModels);
for i = 1:N_LinModels
    ii = L*(i-1) + floor(L*rand(1)) + 1;
    Mu_x(:,i) = X_train(:,ii);
end

% Initialize joint probabilities for calculating the expectation
P_yxuz = zeros(N_train, N_LinModels);
for i = 1:N_LinModels
    for j = 1:N_train
        x = X_train(:,j);
        mu = Mu_x(:,i);
        P_yxuz(j,i) = NormalDist(x,mu,SigInv_Mdls(:,:,i),SigDet_Mdls(i));
    end
end

Q_bar = zeros(N_train, N_LinModels);

%% Fit linear models using EM algorithm

iter = 0;
Delta_Centroids = 1e6*ones(x_dim,N_LinModels);
while any(sqrt(sum(Delta_Centroids.^2,1)) > Tol_centroid) && iter < Iter_Max
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Expectation step
    %
    Sum_P_yxuz = sum(P_yxuz, 2);
    
    for j = 1:N_train
        Q_bar(j,:) = P_yxuz(j,:)/Sum_P_yxuz(j);
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Maximization step
    %
    for k = 1:N_LinModels
        if ones(1,N_train)*Q_bar(:,k) < 1e-10
            Q_bar(:,k) = ones(N_train,1)/N_train;
        end
        
        % Model centroids
        mu_x = X_train*Q_bar(:,k) / (ones(1,N_train)*Q_bar(:,k));
        Delta_Centroids(:,k) = mu_x - Mu_x(:,k);
        Mu_x(:,k) = mu_x;
        
        % Centered data
        Z_k = X_train - Mu_x(:,k)*ones(1,N_train);
        
        % Gaussian covariance
        Sig_k = Z_k*diag(Q_bar(:,k))*Z_k' / (ones(1,N_train)*Q_bar(:,k));
        
        [U,S,V] = svd(Sig_k); % Correct degenerate Gaussians
        eps = eps_Sig*ones(x_dim,1);
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        Sigma_Mdls(:,:,k) = U*S*V';
        SigInv_Mdls(:,:,k) = V*Sinv*U';
        SigDet_Mdls(k) = prod(s);
        
        % Linear models
        Xi_k = [Z_k; U_train; ones(1,N_train)];
%         A_tilde_k = (Y*diag(Q_bar(:,k))*Xi_k') / ...
%             (Xi_k*diag(Q_bar(:,k))*Xi_k');
        
        A_tilde_k = (Y_train*diag(Q_bar(:,k))*Xi_k') * ...
            pinv(Xi_k*diag(Q_bar(:,k))*Xi_k');
        A_matrices(:,:,k) = A_tilde_k(:,1:x_dim);
        B_matrices(:,:,k) = A_tilde_k(:,(x_dim+1):end-1);
        Mu_y(:,k) = A_tilde_k(:,end);
        
        % Find error covariance
        V_k = Y_train - A_tilde_k*Xi_k;
        R_k = V_k*diag(Q_bar(:,k))*V_k' / (ones(1,N_train)*Q_bar(:,k));
        
        [U,S,V] = svd(R_k); % Correct degenerate Gaussians
        eps = eps_R*ones(x_dim,1);
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        R_Mdls(:,:,k) = U*S*V';
        Rinv_Mdls(:,:,k) = V*Sinv*U';
        RDet_Mdls(k) = prod(s);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update joint probabilities for calculating the expectation
        %
        
        % Update priors on each model
        phi_Mdls(k) = 1/N_train * (ones(1,N_train)*Q_bar(:,k));
        
        % Joint probabilities
        for j = 1:N_train
            P_yxuz(j,k) = phi_Mdls(k) * NormalDist(V_k(:,j), ...
                zeros(x_dim,1), Rinv_Mdls(:,:,k), RDet_Mdls(k)) * ...
                NormalDist(X_train(:,j), Mu_x(:,k), SigInv_Mdls(:,:,k), ...
                SigDet_Mdls(k));
        end
    end
    
    iter = iter + 1;
    fprintf('Iteration %d : Max Centroid Change = %.3e \n', ...
        iter, max(sqrt(sum(Delta_Centroids.^2,1))))
end

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

