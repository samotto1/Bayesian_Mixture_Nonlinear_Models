function [LinModels] = MixtureOfKernelMarkovModels_EM( TrainData, Params)
%MixtureOfLinearModels_EM

X_train = TrainData.X;
U_train = TrainData.U;
Y_train = TrainData.Y;

N_Models = Params.N_Models;
N_SubTrain = Params.N_SubTrain;
lambda = Params.lambda;
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

fprintf('\n *** Evaluating kernel... \n')
kernel = Params.kernel;
K = zeros(N_train,N_train);
for i = 1:N_train
    for j = 1:N_train
        z1 = [X_train(:,i); U_train(:,i); 1];
        z2 = [X_train(:,j); U_train(:,j); 1];
        K(i,j) = kernel(z1, z2);
    end
end

%% Initialization
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

for i = 1:N_Models
    Sigma_Mdls(:,:,i) = sig_init*eye(x_dim);
    SigInv_Mdls(:,:,i) = (1.0/sig_init)*eye(x_dim);
    SigDet_Mdls(i) = sig_init^x_dim;
end

% Error covariance of linear models
R_Mdls = zeros(x_dim,x_dim,N_Models);
Rinv_Mdls = zeros(x_dim,x_dim,N_Models);
RDet_Mdls = zeros(1,N_Models);

% Initialize Markov model with uniform distributions
Phi_Mdls = 1/N_Models * ones(N_Models,N_train);
T_Mdls = 1/N_Models * ones(N_Models,N_Models);


% pick initial centroids randomly
L = floor(N_train/N_Models);
for i = 1:N_Models
    ii = L*(i-1) + floor(L*rand(1)) + 1;
    Mu_x(:,i) = X_train(:,ii);
end

% Initialize joint probabilities for calculating the expectation
P_yxuz = zeros(N_train, N_Models, N_Models);
for i = 1:N_Models
    for j = 1:N_train
        x = X_train(:,j);
        mu = Mu_x(:,i);
        P_yxuz(j,i,:) = NormalDist(x,mu,SigInv_Mdls(:,:,i),SigDet_Mdls(i));
    end
end

Q_bar = zeros(N_train, N_Models, N_Models);
q_bar = zeros(N_train, N_Models);

%% Fit linear models using EM algorithm

fprintf('\n *** Beginning EM convergence loop... \n')
iter = 0;
Delta_Centroids = 1e6*ones(x_dim,N_Models);
while any(sqrt(sum(Delta_Centroids.^2,1)) > Tol_centroid) && iter < Iter_Max
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Expectation step
    %
    %Sum_P_yxuz = sum(sum(P_yxuz,3),2);
    
    for j = 1:N_train
        SP = sum(sum(P_yxuz(j,:,:)));
        Q_bar(j,:,:) = P_yxuz(j,:,:)/SP;
    end
    q_bar = sum(Q_bar,3);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Maximization step
    %
    for k = 1:N_Models
%         if ones(1,N_train)*q_bar(:,k) < 1e-10
%             q_bar(:,k) = ones(N_train,1)/N_train;
%         end
        
        % Model centroids
        mu_x = X_train*q_bar(:,k) / (ones(1,N_train)*q_bar(:,k));
        Delta_Centroids(:,k) = mu_x - Mu_x(:,k);
        Mu_x(:,k) = mu_x;
        
        % Centered data
        Z_k = X_train - Mu_x(:,k)*ones(1,N_train);
        
        % Gaussian covariance
        Sig_k = Z_k*diag(q_bar(:,k))*Z_k' / (ones(1,N_train)*q_bar(:,k));
        
        [U,S,V] = svd(Sig_k); % Correct degenerate Gaussians
        eps = eps_Sig*ones(x_dim,1);
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        Sigma_Mdls(:,:,k) = U*S*V';
        SigInv_Mdls(:,:,k) = V*Sinv*U';
        SigDet_Mdls(k) = prod(s);
        
        
        %
        % Nonlinear weighted kernel regression
        %
        
        % choose N_SubTrain points for regression
        [~,ix] = sort(q_bar(:,k), 'descend');
        inds = ix(1:N_SubTrain);
        
        % locate y-centroid
        Mu_y(:,k) = Y_train(:,inds)*q_bar(inds,k) /...
            (ones(1,N_SubTrain)*q_bar(inds,k));
        
        % form training data sub-matrices
        Yc_Sub = Y_train(:,inds) - Mu_y(:,k) * ones(1,N_SubTrain);
        
        % weighted kernel regression
        W = diag(q_bar(inds,k)); % weight matrix
        K_sub = K(inds,inds); % kernel sub-matrix
        M_matrices(:,:,k) = zeros(x_dim,N_train);
        M_matrices(:,inds,k) = Yc_Sub*W*pinv(K_sub*W + ...
            lambda*eye(N_SubTrain));
        
        % Find error covariance
        V_k = Y_train - ( Mu_y(:,k)*ones(1,N_train) + ...
            M_matrices(:,:,k)*K );
        R_k = V_k*diag(q_bar(:,k))*V_k' / (ones(1,N_train)*q_bar(:,k));
        
        [U,S,V] = svd(R_k); % Correct degenerate Gaussians
        eps = eps_R*ones(x_dim,1);
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        R_Mdls(:,:,k) = U*S*V';
        Rinv_Mdls(:,:,k) = V*Sinv*U';
        RDet_Mdls(k) = prod(s);
        
        % update Markov transition matrix
        T_Mdls(:,k) = sum(Q_bar(:,:,k),1) / sum(sum(Q_bar(:,:,k)));
    end
           
	for j = 1:N_train
        
        SQ = sum(sum(Q_bar(j,:,:)));
        for k = 1:N_Models
            % prior probabilities in Markov process
            Phi_Mdls(k,j) = sum(Q_bar(j,:,k)) / SQ;
        end
        
        for k = 1:N_Models
            for i = 1:N_Models
            % Update Joint probabilities
                P_yxuz(j,k,i) = NormalDist(V_k(:,j), zeros(x_dim,1), ...
                    Rinv_Mdls(:,:,k), RDet_Mdls(k)) * ...
                    NormalDist(X_train(:,j), Mu_x(:,k), SigInv_Mdls(:,:,k), ...
                    SigDet_Mdls(k)) * T_Mdls(k,i)*Phi_Mdls(i,j);
            end
        end
	end
    
    iter = iter + 1;
    fprintf('Iteration %d : Max Centroid Change = %.3e \n', ...
        iter, max(sqrt(sum(Delta_Centroids.^2,1))))
end

LinModels.Mu_x = Mu_x;
LinModels.Mu_y = Mu_y;
LinModels.M_matrices = M_matrices;
LinModels.Sigma_Mdls = Sigma_Mdls;
LinModels.SigInv_Mdls = SigInv_Mdls;
LinModels.SigDet_Mdls = SigDet_Mdls;
LinModels.R_Mdls = R_Mdls;
LinModels.Rinv_Mdls = Rinv_Mdls;
LinModels.RDet_Mdls = RDet_Mdls;
LinModels.T_Mdls = T_Mdls;
LinModels.Phi_Mdls = Phi_Mdls;

end

