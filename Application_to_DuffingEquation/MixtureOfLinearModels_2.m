clear 
clc
close all

%%
% ** Simulate Duffing equation with variable combinations of Fourier
% bases as time-dependent forcing.
% ** Seed N_LinModels linear models randomly throughout the data.
% ** Allocate points initially based on distance
% ** Loop through until the centroid locations converge 
%   1. Fit linear models to the clusters
%   2. Assign points to clusters based on minimum error
% ** Fit MLE Gaussians to clusters as a generative model for classification 

%% Paramters
N_LinModels = 50;
N_nearest = 2; % number of nearest models to consider for point assignment
Iter_Max = 100;
Tol_centroid = 1e-6;

N_bases = 0; % number of Fourier bases to use for forcing

% Training and test points
N_train = 2000;
N_test = 1000;

% Domain
pert_p = 1e-3;
Domain_x = [-1.5,1.5];
Domain_y = [-0.6,0.8];

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
Centroids_x = zeros(2,N_LinModels);
% centroids in range
Centroids_y = zeros(2,N_LinModels);
% Linear system matrices
A_matrices = zeros(2,2,N_LinModels);
% Sensitivity matrices
B_matrices = zeros(2,N_bases,N_LinModels);

% Initialize point allocations
MdlCounts = zeros(1,N_LinModels);
MdlIndices = ones(N_LinModels,N_train);

% pick initial centroids randomly
L = floor(N_train/N_LinModels);
for i = 1:N_LinModels
    ii = L*(i-1) + floor(L*rand(1)) + 1;
    Centroids_x(:,i) = X_train(1:2,ii);
end

% allocate points to models initially based on distance
for j = 1:N_train
    x = X_train(1:2,j);
    Dx = x*ones(1,N_LinModels) - Centroids_x;
    dists = sqrt(sum(Dx.^2,1));
    [~,idx_mdl] = min(dists);
    
    MdlCounts(idx_mdl) = MdlCounts(idx_mdl) + 1;
    MdlIndices(idx_mdl, MdlCounts(idx_mdl)) = j;
end

% Plot the different points
figure(2)
C = lines(N_LinModels);
for i = 1:N_LinModels
    indices = MdlIndices(i,1:MdlCounts(i));
    hold on
    plot(X_train(1,indices), X_train(2,indices), '.', ...
        'Color', C(i,:), 'LineWidth',1.5)
    plot(Centroids_x(1,i), Centroids_x(2,i), 'x', ...
        'Color', C(i,:), 'LineWidth',2)
end
hold off
title('Initial Clusters and Centroids')
drawnow

%% Fit mixture of linear models
Mdl_error = zeros(1,N_nearest);
iter = 0;
Delta_Centroids = 1e6*ones(2,N_LinModels);
while any(sqrt(sum(Delta_Centroids.^2,1)) > Tol_centroid) && iter < Iter_Max
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Locate centroids and fit linear models
    %
    for i = 1:N_LinModels
        % indices belonging to cluster
        indices = MdlIndices(i,1:MdlCounts(i));
        
        % data points in cluster
        X_cluster = X_train(:, indices);
        Y_cluster = Y_train(:, indices);
        
        % calculate centroids
        Centroid_x_new = mean(X_cluster(1:2,:),2);
        Delta_Centroids(:,i) = Centroid_x_new - Centroids_x(:,i);
        Centroids_x(:,i) = Centroid_x_new;
        Centroids_y(:,i) = mean(Y_cluster(1:2,:),2);
        
        % fit linear model
        DX = [X_cluster(1:2,:) - Centroids_x(:,i)*ones(1,MdlCounts(i));
            X_cluster(3:end,:)];
        DY = Y_cluster - Centroids_y(:,i)*ones(1,MdlCounts(i));
        M = DY*pinv(DX);
        A_matrices(:,:,i) = M(:,1:2);
        B_matrices(:,:,i) = M(:,3:end);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Assign points to models based on maximum likelihood
    %
    
    % First empty the previous point assignments
    MdlCounts = zeros(1,N_LinModels);
    MdlIndices = ones(N_LinModels,N_train);
    
    % Assign points to models
    for j = 1:N_train
        x = X_train(1:2,j);
        p = X_train(3:end,j);
        y = Y_train(1:2,j);
        
        % Locate the N_nearest nearest models
        Dx = x*ones(1,N_LinModels) - Centroids_x;
        Dist = sqrt(sum(Dx.^2,1));
        [~,inds] = sort(Dist);
        MdlInds = inds(1:N_nearest);
        
        % Evaluate errors using each model 
        for ii = 1:N_nearest
            idx_mdl = MdlInds(ii);
            
            y_hat = A_matrices(:,:,idx_mdl)*(x - Centroids_x(:,idx_mdl)) + ...
                B_matrices(:,:,idx_mdl)*p + Centroids_y(:,idx_mdl);
            y_error = y - y_hat;
            
            Mdl_error(ii) = sqrt(y_error'*y_error);
        end
        
        % Assign to model based on minimum error
        [~, ii_star] = min(Mdl_error);
        idx_mdl_star = MdlInds(ii_star);
        MdlCounts(idx_mdl_star) = MdlCounts(idx_mdl_star) + 1;
        MdlIndices(idx_mdl_star, MdlCounts(idx_mdl_star)) = j;
    end
    
    iter = iter + 1;
    fprintf('Iteration %d : Max Centroid Change = %.3e \n', ...
        iter, max(sqrt(sum(Delta_Centroids.^2,1))))
end

%% Fit MLE Gaussians to clusters as a generative model for classification

Sigma_Mdls = zeros(2,2,N_LinModels);
SigInv_Mdls = zeros(2,2,N_LinModels);
SigDet_Mdls = zeros(1,N_LinModels);

for i = 1:N_LinModels
    % indices belonging to cluster
	indices = MdlIndices(i,1:MdlCounts(i));
        
	% data points in cluster
	Xpts_cluster = X_train(1:2, indices);
    DXpts = Xpts_cluster - Centroids_x(:,i)*ones(1,MdlCounts(i));
    
    % MLE covariance
    Sig = 1.0/MdlCounts(i) *(DXpts*DXpts');
    [U,S,V] = svd(Sig);
    eps = 1e-6*[1;1];
    s = max([diag(S), eps], [], 2);
    S = diag(s);
    sinv = 1./s;
    Sinv = diag(sinv);
    
    Sigma_Mdls(:,:,i) = U*S*V';
    SigInv_Mdls(:,:,i) = V*Sinv*U';
    SigDet_Mdls(i) = s(1)*s(2);
end


%% Plot point assignments in train data

% Assign points to models
P_Mdls = zeros(1,N_LinModels);
for j = 1:N_train
    x = X_train(1:2,j);
    
    % Evaluate likelihood each model
    for i = 1:N_LinModels
        Dx = x - Centroids_x(:,i);
        
        P_Mdls(i) = 1.0/((2*pi) * sqrt(SigDet_Mdls(i))) *...
                exp(-0.5 * Dx'*SigInv_Mdls(:,:,i)*Dx);
        
    end
    
    % Assign to model based on max likelihood
    [~, i_star] = max(P_Mdls);
    MdlCounts(i_star) = MdlCounts(i_star) + 1;
    MdlIndices(i_star, MdlCounts(i_star)) = j;
end

% Plot the different points
figure(3)
C = lines(N_LinModels);
for i = 1:N_LinModels
    indices = MdlIndices(i,1:MdlCounts(i));
    hold on
    plot(X_train(1,indices), X_train(2,indices), '.', ...
        'Color', C(i,:), 'LineWidth',1.5)
    plot(Centroids_x(1,i), Centroids_x(2,i), 'x', ...
        'Color', C(i,:), 'LineWidth',2)
end
hold off
title('Final Clusters and Centroids')
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
        Dx = x - Centroids_x(:,i);
        
        P_Mdls(i) = 1.0/((2*pi) * sqrt(SigDet_Mdls(i))) *...
                exp(-0.5 * Dx'*SigInv_Mdls(:,:,i)*Dx);
        
    end
    
    % Assign to model based on max likelihood
    [~, idx_mdl] = max(P_Mdls);
    
    % Prediction
    Y_hat(:,j) = A_matrices(:,:,idx_mdl)*(x - Centroids_x(:,idx_mdl)) + ...
                B_matrices(:,:,idx_mdl)*p + Centroids_y(:,idx_mdl);
end

% Error
Y_error = Y_hat - Y_test;
Dist_error = sqrt(sum(Y_error.^2,1));
MSE = mean(sum(Y_error.^2,1))
RMSE = sqrt(MSE)

figure()
nbins = 5*sqrt(N_test);
hist(Dist_error, nbins)
xlabel('Trajectory Error Magnitude')
ylabel('Number of Test Points')
title('Distribution of Testing Error Magnitudes')
