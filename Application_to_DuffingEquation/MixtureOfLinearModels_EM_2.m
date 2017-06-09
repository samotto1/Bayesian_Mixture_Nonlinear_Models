clear 
clc
close all

%%
% ** Simulate Duffing equation with variable combinations of Fourier
% bases as time-dependent forcing.
% ** Seed N_LinModels linear models randomly throughout the data.
% ** use EM algorithm to fit a mixture of linear models with Gaussian
% densities
%   -- Use maximum likelihood Q
% ** Use MLE Gaussians as a generative model for classification

%% Paramters
N_LinModels = 40;
N_ML = 10; % number of max likelihood models to consider at point assignment
Iter_Max = 100;
Tol_centroid = 1e-4;

sig_init = 0.1;

N_bases = 5; % number of Fourier bases to use for forcing

% Training and test points
N_train = 2000;
N_test = 1000;

% Domain
pert_p = 1e-3;
Domain_x = [-1.5,1.5];
Domain_y = [-0.6,0.8];

NormalDist = @(x, mu, SigInv, SigDet) ...
    1/( (2*pi)^(length(x)/2)*sqrt(SigDet) ) *...
    exp( -0.5*(x-mu)'*SigInv*(x-mu) );

%% Define ODE (Forced Duffing Equation)

% Duffing Parameters
alpha = -1;
beta = 1;
delta = 0.25;
gamma = 0.30;
omega = 1.0;

omega_0 = omega;
T = 2*pi/omega;

% Construct basis vectors
N_b_cos = ceil(N_bases/2); % number of cosine terms
N_b_sin = N_bases-N_b_cos; % number of sin terms
bi = @(t,i) (i<=N_b_cos).*cos((i-1)*omega_0*t) + ...
    (i>N_b_cos).*sin((i-N_b_cos)*omega_0*t); % basis functions

% find Gram matrix for basis functions
weight_basis = @(t) 1.0; %Optional weight function along trajectory
Gw_bi = integral( @(t) weight_basis(t)*bi(t,(1:N_bases)')*bi(t,(1:N_bases)), ...
    0, T, 'ArrayValued', true);

% Duffing ODE with additional forcing input
ode_fun = @(x,t,forcing) [  x(2);
              	gamma*cos(omega*t) - delta*x(2) - alpha*x(1) - ...
                beta*x(1).^3 + forcing];
            
%% Generate training data

x_IC = [0;0];
ops = odeset('RelTol', 1e-6);

X_train = zeros(2+N_bases,N_train);
Y_train = zeros(2,N_train);
for j = 1:N_train
    % random time-dependent forcing
    p = pert_p*(2*rand(N_bases,1) - 1);
    f = @(t) bi(t,1:N_bases) * p;
    
    X_train(1:2,j) = x_IC;
    X_train(3:end,j) = p;
    
    % run simulation
    [~,Y] = ode45(@(t,x) ode_fun(x,t,f(t)), [0,T], x_IC, ops);
    Y_train(:,j) = Y(end,:)';
    
    % update
    x_IC = Y(end,:)';
end

figure(1)
f1_p1 = plot(Y_train(1,:), Y_train(2,:), 'k.');
title('Training Data')
drawnow

%% Initialize mixture of linear models

% centroids in domain
Mu_x = zeros(2,N_LinModels);
% centroids in range
Mu_y = zeros(2,N_LinModels);
% Linear system matrices
A_matrices = zeros(2,2,N_LinModels);
% Sensitivity matrices
B_matrices = zeros(2,N_bases,N_LinModels);

% Covariance of MLE Gaussians
Sigma_Mdls = zeros(2,2,N_LinModels);
SigInv_Mdls = zeros(2,2,N_LinModels);
SigDet_Mdls = zeros(1,N_LinModels);

for i = 1:N_LinModels
    Sigma_Mdls(:,:,i) = sig_init*eye(2);
    SigInv_Mdls(:,:,i) = (1.0/sig_init)*eye(2);
    SigDet_Mdls(i) = sig_init*sig_init;
end

% Error covariance of linear models
R_Mdls = zeros(2,2,N_LinModels);
Rinv_Mdls = zeros(2,2,N_LinModels);
RDet_Mdls = zeros(1,N_LinModels);

% Prior likelihoods of models
phi_Mdls = ones(1,N_LinModels)/N_LinModels;

% pick initial centroids randomly
L = floor(N_train/N_LinModels);
for i = 1:N_LinModels
    ii = L*(i-1) + floor(L*rand(1)) + 1;
    Mu_x(:,i) = X_train(1:2,ii);
end

% Initialize joint probabilities for calculating the expectation
P_yxuz = zeros(N_train, N_LinModels);
for i = 1:N_LinModels
    for j = 1:N_train
        x = X_train(1:2,j);
        mu = Mu_x(:,i);
        P_yxuz(j,i) = NormalDist(x,mu,SigInv_Mdls(:,:,i),SigDet_Mdls(i));
    end
end

Q_bar = zeros(N_train, N_LinModels);

%% Fit mixture of linear models using the EM algorithm
iter = 0;
Delta_Centroids = 1e6*ones(2,N_LinModels);
while any(sqrt(sum(Delta_Centroids.^2,1)) > Tol_centroid) && iter < Iter_Max
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Expectation step
    %
    
    % choose only N_ML Gaussians with maximum likelihood at each point
    for j = 1:N_train
        [Pvals, inds] = sort(P_yxuz(j,:),'descend');
        P_yxuz(j,:) = zeros(1,N_LinModels);
        P_yxuz(j,inds(1:N_ML)) = max([Pvals(1:N_ML); 1e-6*ones(1,N_ML)], [], 1);
    end
    
    Q_bar = P_yxuz ./ (P_yxuz*ones(N_LinModels,N_LinModels));
    inds = find(isinf(Q_bar));
    Q_bar(inds) = 0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Maximization step
    %
    for k = 1:N_LinModels
        % Model centroids
        mu_x = X_train(1:2,:)*Q_bar(:,k) / (ones(1,N_train)*Q_bar(:,k));
        Delta_Centroids(:,k) = mu_x - Mu_x(:,k);
        Mu_x(:,k) = mu_x;
        
        % Centered data
        Z_k = X_train(1:2,:) - Mu_x(:,k)*ones(1,N_train);
        
        % Gaussian covariance
        Sig_k = Z_k*diag(Q_bar(:,k))*Z_k' / (ones(1,N_train)*Q_bar(:,k));
        
        [U,S,V] = svd(Sig_k); % Correct degenerate Gaussians
        eps = 1e-10*[1;1];
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        Sigma_Mdls(:,:,k) = U*S*V';
        SigInv_Mdls(:,:,k) = V*Sinv*U';
        SigDet_Mdls(k) = s(1)*s(2);
        
        % Linear models
        Xi_k = [Z_k; X_train(3:end,:); ones(1,N_train)];
        A_tilde_k = (Y_train*diag(Q_bar(:,k))*Xi_k') / ...
            (Xi_k*diag(Q_bar(:,k))*Xi_k');
        A_matrices(:,:,k) = A_tilde_k(:,1:2);
        B_matrices(:,:,k) = A_tilde_k(:,3:end-1);
        Mu_y(:,k) = A_tilde_k(:,end);
        
        % Find error covariance
        V_k = Y_train - A_tilde_k*Xi_k;
        R_k = V_k*diag(Q_bar(:,k))*V_k' / (ones(1,N_train)*Q_bar(:,k));
        
        [U,S,V] = svd(R_k); % Correct degenerate Gaussians
        eps = 1e-10*[1;1];
        s = max([diag(S), eps], [], 2);
        S = diag(s);
        sinv = 1./s;
        Sinv = diag(sinv);
        
        R_Mdls(:,:,k) = U*S*V';
        Rinv_Mdls(:,:,k) = V*Sinv*U';
        RDet_Mdls(k) = s(1)*s(2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update joint probabilities for calculating the expectation
        %
        
        % Update priors on each model
        phi_Mdls(k) = 1/N_train * (ones(1,N_train)*Q_bar(:,k));
        
        % Joint probabilities
        for j = 1:N_train
            P_yxuz(j,k) = phi_Mdls(k) * NormalDist(V_k(:,j), ...
                zeros(2,1), Rinv_Mdls(:,:,k), RDet_Mdls(k)) * ...
                NormalDist(X_train(1:2,j), Mu_x(:,k), SigInv_Mdls(:,:,k), ...
                SigDet_Mdls(k));
        end
    end
    
    
    iter = iter + 1;
    fprintf('Iteration %d : Max Centroid Change = %.3e \n', ...
        iter, max(sqrt(sum(Delta_Centroids.^2,1))))
end


%% Assign points to models and Plot point assignments in train data

% Initialize point allocations
MdlCounts = zeros(1,N_LinModels);
MdlIndices = ones(N_LinModels,N_train);

% Assign points to models
P_Mdls = zeros(1,N_LinModels);
for j = 1:N_train
    x = X_train(1:2,j);
    
    % Evaluate likelihood each model
    for i = 1:N_LinModels
        
        P_Mdls(i) = phi_Mdls(i) * NormalDist(x, Mu_x(:,i), ...
            SigInv_Mdls(:,:,i), SigDet_Mdls(i));
        
    end
    
    % Assign to model based on max likelihood
    [~, i_star] = max(P_Mdls);
    MdlCounts(i_star) = MdlCounts(i_star) + 1;
    MdlIndices(i_star, MdlCounts(i_star)) = j;
end

% Plot the different points
figure(2)
C = lines(N_LinModels);
for i = 1:N_LinModels
    indices = MdlIndices(i,1:MdlCounts(i));
    hold on
    plot(X_train(1,indices), X_train(2,indices), '.', ...
        'Color', C(i,:), 'LineWidth',1.5)
    plot(Mu_x(1,i), Mu_x(2,i), 'x', ...
        'Color', C(i,:), 'LineWidth',2)
end
hold off
title('Final EM Clusters and Centroids')
drawnow

%% Generate testing data

% random initial condition in domain
x_IC = [Domain_x(1);Domain_y(1)] +...
    [Domain_x(2)-Domain_x(1);Domain_y(2)-Domain_y(1)].*rand(2,1);
ops = odeset('RelTol', 1e-6);

X_test = zeros(2+N_bases,N_test);
Y_test = zeros(2,N_test);
for j = 1:N_test
    % random time-dependent forcing
    p = pert_p*(2*rand(N_bases,1) - 1);
    f = @(t) bi(t,1:N_bases) * p;
    
    X_test(1:2,j) = x_IC;
    X_test(3:end,j) = p;
    
    % run simulation
    [~,Y] = ode45(@(t,x) ode_fun(x,t,f(t)), [0,T], x_IC, ops);
    Y_test(:,j) = Y(end,:)';
    
    % update
    x_IC = Y(end,:)';
end

%% Determine error on the testing data

Y_hat = zeros(2,N_test);
P_Mdls = zeros(1,N_LinModels);
for j = 1:N_test
    x = X_test(1:2,j);
    p = X_test(3:end,j);
    
    % Evaluate likelihood each model
    for i = 1:N_LinModels
        
        P_Mdls(i) = phi_Mdls(i) * NormalDist(x, Mu_x(:,i), ...
            SigInv_Mdls(:,:,i), SigDet_Mdls(i));
        
    end
    
    % Assign to model based on max likelihood
    [~, idx_mdl] = max(P_Mdls);
    
    % Prediction
    Y_hat(:,j) = A_matrices(:,:,idx_mdl)*(x - Mu_x(:,idx_mdl)) + ...
                B_matrices(:,:,idx_mdl)*p + Mu_y(:,idx_mdl);
end

% Error
Y_error = Y_hat - Y_test;
Dist_error = sqrt(sum(Y_error.^2,1));
MSE = mean(sum(Y_error.^2,1))
RMSE = sqrt(MSE)

DistError_Mean = mean(Dist_error)
DistError_Median = median(Dist_error)

figure()
nbins = 5*sqrt(N_test);
bin_edges = linspace(0,0.25,26);
bins = 0.5*(bin_edges(1:end-1) + bin_edges(2:end));
hist(Dist_error, bins)
xlabel('Trajectory Error Magnitude')
ylabel('Number of Test Points')
title('Distribution of Testing Error Magnitudes')
xlim([0,0.25])