clear 
clc
close all

%%
% ** Simulate Duffing equation with variable combinations of Fourier
% bases as time-dependent forcing.
% ** Seed N_LinModels nonlinear kernel models randomly throughout the data.
% ** use EM algorithm to fit a mixture of regularized kernel models with
% Gaussian densities
% ** Use MLE Gaussians as a generative model for classification

%% Paramters
N_Models = 20;
N_SubTrain = 500; % number of points to use in Kernel regression
lambda = 0.0005; % Ridge regression regularization parameter
Iter_Max = 200;
Tol_centroid = 1e-4;

%kernel = @(x,z) (x'*z + 1).^4;
kernel = @(x,z) exp(-1/2 * (x-z)'*(x-z) / 0.1^2);

% minimum allowable singular values of Spatial and Error covariance
% matrices used to prevent degenerate Gaussians.
eps_Sig = 1e-4;
eps_R = 1e-4; 

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

%% Fit mixture of nonlinear models using kernel regression and EM

% assign training data
TrainData.X = X_train(1:2,:);
TrainData.U = X_train(3:end,:);
TrainData.Y = Y_train;

% assign parameters
Params.N_Models = N_Models;
Params.N_SubTrain = N_SubTrain;
Params.lambda = lambda;
Params.Iter_Max = Iter_Max;
Params.Tol_centroid = Tol_centroid;
Params.eps_Sig = eps_Sig;
Params.eps_R = eps_R;

Params.sig_init = sig_init;

Params.kernel = kernel;

% EM algorithm with kernel ridge regression
[LinModels] = MixtureOfKernelModels_EM( TrainData, Params);

% converged model parameters
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

%% Apply models to classify testing data and estimate dynamics

% Initialize point allocations based on maximum likelihood
MdlCounts_MLE = zeros(1,N_Models);
MdlIndices_MLE = ones(N_Models,N_test);

%Initialize point allocations based on stochastic model choice
MdlCounts_Stoch = zeros(1,N_Models);
MdlIndices_Stoch = ones(N_Models,N_test);


% Initialize estimates
Y_hat_MLE = zeros(2,N_test); % using maximum likelihood model
Y_hat_Mix = zeros(2,N_test); % using weighted mixture of all models
Y_hat_Stoch = zeros(2,N_test); % using stochastically chosen model

% Assign points to models and construct estimates
W_Mdls = zeros(1,N_Models);
K_z = zeros(N_train,1);
for j = 1:N_test
    x = X_test(1:2,j);
    u = X_test(3:end,j);
    z = [x;u;1];
    
    % Evaluate likelihood each model
    for i = 1:N_Models
        W_Mdls(i) = phi_Mdls(i) * NormalDist(x, Mu_x(:,i), ...
            SigInv_Mdls(:,:,i), SigDet_Mdls(i)); 
    end
    W_Mdls = W_Mdls/sum(W_Mdls);
    
    % Assign to model based on max likelihood
    [~, i_star] = max(W_Mdls);
    MdlCounts_MLE(i_star) = MdlCounts_MLE(i_star) + 1;
    MdlIndices_MLE(i_star, MdlCounts_MLE(i_star)) = j;
    
    % Assign to model based on stochastic choice
    r= rand(1);
    csumW = cumsum(W_Mdls);
    i_stoch = find(csumW >= r, 1, 'first');
    MdlCounts_Stoch(i_stoch) = MdlCounts_Stoch(i_stoch) + 1;
    MdlIndices_Stoch(i_stoch, MdlCounts_Stoch(i_stoch)) = j;
    
    % Evaluate kernel at test point
    for l = 1:N_train
        K_z(l) = kernel([X_train(:,l);1],z);
    end
    
    % MLE model estimate
    Y_hat_MLE(:,j) = M_matrices(:,:,i_star)*K_z + Mu_y(:,i_star);
    
    % Weighted mixture of models estimate
    for i = 1:N_Models      
        Y_hat_Mix(:,j) =Y_hat_Mix(:,j) + W_Mdls(i)*...
            ( M_matrices(:,:,i)*K_z + Mu_y(:,i) );     
    end
    
    % Stochastic model estimate
    Y_hat_Stoch(:,j) = M_matrices(:,:,i_stoch)*K_z + Mu_y(:,i_stoch);
end

% Plot testing point assignments using MLE
figure(2)
C = lines(N_Models);
for i = 1:N_Models
    indices = MdlIndices_MLE(i,1:MdlCounts_MLE(i));
    hold on
    plot(X_test(1,indices), X_test(2,indices), '.', ...
        'Color', C(i,:), 'LineWidth',1.5)
    plot(Mu_x(1,i), Mu_x(2,i), 'x', ...
        'Color', C(i,:), 'LineWidth',2)
end
hold off
title({'Final EM Clusters and Centroids',...
    'Model Assignment using Maximum Likelihood'})
drawnow

% Plot testing point assignments using stochstic assignment
figure(3)
C = lines(N_Models);
for i = 1:N_Models
    indices = MdlIndices_Stoch(i,1:MdlCounts_Stoch(i));
    hold on
    plot(X_test(1,indices), X_test(2,indices), '.', ...
        'Color', C(i,:), 'LineWidth',1.5)
    plot(Mu_x(1,i), Mu_x(2,i), 'x', ...
        'Color', C(i,:), 'LineWidth',2)
end
hold off
title({'Final EM Clusters and Centroids',...
    'Stochastic Model Assignment'})
drawnow

%% Quantify errors on training data

% Error using MLE model
fprintf('\n *** MLE Model Assignment Error: *** \n')
Y_error = Y_hat_MLE - Y_test;
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
title({'Distribution of Testing Error Magnitudes',...
    'using Maximum Likelihood Model Assignment'})
xlim([0,0.25])
drawnow

% Error using weighted mixture of models
fprintf('\n *** Weighted Mixture of Models Error: *** \n')
Y_error = Y_hat_Mix - Y_test;
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
title({'Distribution of Testing Error Magnitudes',...
    'using Weighted Mixture of Models'})
xlim([0,0.25])
drawnow

% Error using weighted mixture of models
fprintf('\n *** Stochastic Model Assignment Error: *** \n')
Y_error = Y_hat_Stoch - Y_test;
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
title({'Distribution of Testing Error Magnitudes',...
    'using Stochastic Model Assignment'})
xlim([0,0.25])
drawnow

%% Generate new data using model
N_generate = 1000;
X_IC = [0.5902;0.5247];
U_gen = zeros(N_bases,N_generate);

X_gen = zeros(2,N_generate);
X_gen(:,1) = X_IC;

MdlCounts_gen = zeros(1,N_Models);
MdlIndices_gen = ones(N_Models,N_generate);

for n = 2:N_generate
    x = X_gen(:,n-1);
    u = U_gen(:,n-1);
    z = [x;u;1];
    
    % Evaluate likelihood each model
    for i = 1:N_Models
        W_Mdls(i) = phi_Mdls(i) * NormalDist(x, Mu_x(:,i), ...
            SigInv_Mdls(:,:,i), SigDet_Mdls(i)); 
    end
    W_Mdls = W_Mdls/sum(W_Mdls);
    
    % Assign to model based on max likelihood
    [~, i_star] = max(W_Mdls);
    
    % Assign to model based on stochastic choice
    r= rand(1);
    csumW = cumsum(W_Mdls);
    i_stoch = find(csumW >= r, 1, 'first');
    
    % Evaluate kernel at test point
    for l = 1:N_train
        K_z(l) = kernel([X_train(:,l);1],z);
    end
    
    % MLE model estimate
    %X_gen(:,n) = M_matrices(:,:,i_star)*K_z + Mu_y(:,i_star);
    
    MdlCounts_gen(i_star) = MdlCounts_gen(i_star) + 1;
    MdlIndices_gen(i_star, MdlCounts_gen(i_star)) = n-1;
    
    % Weighted mixture of models estimate
   for i = 1:N_Models      
       X_gen(:,n) = X_gen(:,n) + W_Mdls(i)*...
           ( M_matrices(:,:,i)*K_z + Mu_y(:,i) );     
   end
%     
%     % Stochastic model estimate
     %X_gen(:,n) = M_matrices(:,:,i_stoch)*K_z + Mu_y(:,i_stoch);
end

figure()
C = lines(N_Models);
for i = 1:N_Models
    indices = MdlIndices_gen(i,1:MdlCounts_gen(i));
    hold on
    plot(X_gen(1,indices), X_gen(2,indices), '.', ...
        'Color', C(i,:), 'LineWidth',1.5)
    plot(Mu_x(1,i), Mu_x(2,i), 'x', ...
        'Color', C(i,:), 'LineWidth',2)
end
hold off
title({'Dynamics using Generative Model'})
xlim([-1.5,1.5])
ylim([-0.6,0.8])
grid on
drawnow

%% Look at cross-correlation

% generate un-forced data
x_IC = [0.5902;0.5247];
ops = odeset('RelTol', 1e-6);

X_data = zeros(2+N_bases,N_generate);
Y_data = zeros(2,N_generate);
for j = 1:N_generate
    % random time-dependent forcing
    p = 0*(2*rand(N_bases,1) - 1);
    f = @(t) bi(t,1:N_bases) * p;
    
    X_data(1:2,j) = x_IC;
    X_data(3:end,j) = p;
    
    % run simulation
    [~,Y] = ode45(@(t,x) ode_fun(x,t,f(t)), [0,T], x_IC, ops);
    Y_data(:,j) = Y(end,:)';
    
    % update
    x_IC = Y(end,:)';
end

% find cross-correlations
N_lag = 15;

[corr_xx_data, lags] = xcorr(X_data(1,:), X_data(1,:), N_lag, 'coeff');
[corr_xy_data, ~] = xcorr(X_data(1,:), X_data(2,:), N_lag, 'coeff');
[corr_yy_data, ~] = xcorr(X_data(2,:), X_data(2,:), N_lag, 'coeff');

[corr_xx_gen, ~] = xcorr(X_gen(1,:), X_gen(1,:), N_lag, 'coeff');
[corr_xy_gen, ~] = xcorr(X_gen(1,:), X_gen(2,:), N_lag, 'coeff');
[corr_yy_gen, ~] = xcorr(X_gen(2,:), X_gen(2,:), N_lag, 'coeff');

figure()
subplot(3,1,1)
p1 = stairs(lags(N_lag+1:end), corr_xx_data(N_lag+1:end), 'k-', 'LineWidth', 1.5);
hold on
p2 = stairs(lags(N_lag+1:end), corr_xx_gen(N_lag+1:end), 'b-', 'LineWidth', 1.5);
hold off
grid on; ylim([-1,1]);
title('n-back Cross-Correlations in Data')
ylabel('E[x(j)x(j-n)]')
legend([p1,p2], {'Duffing Data', 'Generative Model'})

subplot(3,1,2)
stairs(lags(N_lag+1:end), corr_xy_data(N_lag+1:end), 'k-', 'LineWidth', 1.5);
hold on
stairs(lags(N_lag+1:end), corr_xy_gen(N_lag+1:end), 'b-', 'LineWidth', 1.5);
hold off
grid on
ylabel('E[x(j)y(j-n)]')

subplot(3,1,3)
stairs(lags(N_lag+1:end), corr_yy_data(N_lag+1:end), 'k-', 'LineWidth', 1.5);
hold on
stairs(lags(N_lag+1:end), corr_yy_gen(N_lag+1:end), 'b-', 'LineWidth', 1.5);
hold off
grid on
ylabel('E[y(j)y(j-n)]')
xlabel('Lag, n')