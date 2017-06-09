clear 
clc
close all

%%
% ** Simulate Duffing equation with variable combinations of Fourier
% bases as time-dependent forcing.
% ** Seed N_LinModels linear models randomly throughout the data.
% ** Allocate points initially based on distance
% ** Loop through models until the centroid locations converge 
%   1. Fit MLE Gaussians to point clusters
%   2. Fit linear models to clusters using Gaussians for weighting
%   3. Assign points to clusters based on minimum error
% ** Use MLE Gaussians as a generative model for classification

%% Paramters
N_LinModels = 40;
N_nearest = 2; % number of nearest models to consider for point assignment
Iter_Max = 100;
Tol_centroid = 1e-10;

eps_Sig = 1e-6;
eps_R = 1e-6;

N_bases = 5; % number of Fourier bases to use for forcing

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


%% Fit mixture of linear models

TrainData.X = X_train(1:2,:);
TrainData.U = X_train(3:end,:);
TrainData.Y = Y_train;

Params.N_Models = N_LinModels;
Params.N_Nearest = N_nearest;
Params.Iter_Max = Iter_Max;
Params.Tol_centroid = Tol_centroid;
Params.eps_Sig = eps_Sig;
Params.eps_R = eps_R;

LinModels = MixtureOfLinearModels_Kmeans( TrainData, Params);


Centroids_x = LinModels.Mu_x;
Centroids_y = LinModels.Mu_y;
A_matrices = LinModels.A_matrices;
B_matrices = LinModels.B_matrices;
Sigma_Mdls  = LinModels.Sigma_Mdls;
SigInv_Mdls = LinModels.SigInv_Mdls;
SigDet_Mdls = LinModels.SigDet_Mdls;
Phi_Mdls = LinModels.phi_Mdls;

MdlCounts = LinModels.MdlCounts;
MdlIndices = LinModels.MdlIndices;


%% Assign points to models and Plot point assignments in train data

% Assign points to models
P_Mdls = zeros(1,N_LinModels);
for j = 1:N_train
    x = X_train(1:2,j);
    
    % Evaluate likelihood each model
    for i = 1:N_LinModels
        Dx = x - Centroids_x(:,i);
        
        P_Mdls(i) = 1.0/((2*pi) * sqrt(SigDet_Mdls(i))) *...
                exp(-0.5 * Dx'*SigInv_Mdls(:,:,i)*Dx) * Phi_Mdls(i);
        
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
                exp(-0.5 * Dx'*SigInv_Mdls(:,:,i)*Dx) * Phi_Mdls(i);
        
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