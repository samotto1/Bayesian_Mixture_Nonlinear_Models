clear; clc; close all;

fname_TrainData = 'DuffingTrainData.mat';
fname_TestData = 'DuffingTestData.mat';

fname_ModelFit = 'DuffingKernelModelsFit.mat';

%% Paramters
N_Models = 40;
TopN = [1,4]; % top models to choose from for each point
lambda = 0.0005; % Ridge regression regularization parameter
Iter_Max = 200;
Tol_centroid = [1e-4, 1e-6];

%kernel = @(x,z) x'*z;
%kernel = @(x,z) (x'*z + 1).^4;
kernel = @(x,z) exp(-1/2 * (x-z)'*(x-z) / 0.1^2);

Dz_kernel = @(x,z) 1/0.1^2 * exp(-1/2 * (x-z)'*(x-z) / 0.1^2) * (x-z)';

% minimum allowable singular values of Spatial and Error covariance
% matrices used to prevent degenerate Gaussians.
eps_Sig = [1e-4,1e-4];
eps_R = [1e-4,1e-4];

%% Fit mixture of nonlinear models using kernel regression and EM/K-means

TrainDataFile = load(fname_TrainData);
N_train = TrainDataFile.N_samples;

% assign training data
TrainData.X = TrainDataFile.X_data;
TrainData.U = TrainDataFile.U_data;
TrainData.Y = TrainDataFile.Y_data;

% assign parameters
Params.N_Models = N_Models;
Params.TopN = TopN(1);
Params.lambda = lambda;
Params.Iter_Max = Iter_Max;
Params.Tol_centroid = Tol_centroid(1);
Params.eps_Sig = eps_Sig(1);
Params.eps_R = eps_R(1);

Params.kernel = kernel;

% use random initialization first
Model_Init.init_yn = 'n';

% EM algorithm with kernel ridge regression
[KerModels] = MixtureOfKernelModels_Kmeans_TopN( TrainData, Params,...
    Model_Init);

% Increase number of models allocated to each point
Params.TopN = TopN(2);
Params.Tol_centroid = Tol_centroid(2);
Params.eps_Sig = eps_Sig(2);
Params.eps_R = eps_R(2);

fprintf('\n Resuming with TopN = %d models \n', Params.TopN);
Model_Init = KerModels;

% use previous fit for initialization
Model_Init.init_yn = 'y';
[KerModels] = MixtureOfKernelModels_Kmeans_TopN( TrainData, Params,...
    Model_Init);

% converged model parameters
Mu_x = KerModels.Mu_x;
Mu_y = KerModels.Mu_y;
M_matrices = KerModels.M_matrices;
Sigma_Mdls = KerModels.Sigma_Mdls;
SigInv_Mdls = KerModels.SigInv_Mdls;
SigDet_Mdls = KerModels.SigDet_Mdls;
R_Mdls = KerModels.R_Mdls;
Rinv_Mdls = KerModels.Rinv_Mdls;
RDet_Mdls = KerModels.RDet_Mdls;
phi_Mdls = KerModels.phi_Mdls;

%% Save Models to File
Gw_bi = TrainDataFile.Gw_bi;
save(fname_ModelFit, 'KerModels', 'TrainData', 'kernel', 'Dz_kernel',...
    'Gw_bi');

%% Evaluate Models on Testing Data
clear TrainDataFile
TestDataFile = load(fname_TestData);

N_test = TestDataFile.N_samples;
X_test = TestDataFile.X_data;
U_test = TestDataFile.U_data;
Y_test = TestDataFile.Y_data;

% Initialize point allocations based on maximum likelihood
MdlCounts_MLE = zeros(1,N_Models);
MdlIndices_MLE = ones(N_Models,N_test);

% Initialize estimates
Y_hat_MLE = zeros(2,N_test); % using maximum likelihood model

% Assign points to models and construct estimates
W_Mdls = zeros(1,N_Models);
K_z = zeros(N_train,1);
for j = 1:N_test
    x = X_test(:,j);
    u = U_test(:,j);
    z = [x;u;1];
    
    [ y_hat, i_star ] = MLE_KernelModel_Eval( z, kernel, KerModels,...
        TrainData );
    
    % Assign to model based on max likelihood
    MdlCounts_MLE(i_star) = MdlCounts_MLE(i_star) + 1;
    MdlIndices_MLE(i_star, MdlCounts_MLE(i_star)) = j;
    
    % MLE model estimate
    Y_hat_MLE(:,j) = y_hat;
    
end

% Plot testing point assignments using MLE
figure()
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
grid on
title({'Final EM Clusters and Centroids',...
    'Model Assignment using Maximum Likelihood'})
drawnow

%% Quantify Errors on Testing Data

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
bin_edges = linspace(0,0.05,26);
bins = 0.5*(bin_edges(1:end-1) + bin_edges(2:end));
hist(Dist_error, bins)
xlabel('Trajectory Error Magnitude')
ylabel('Number of Test Points')
title({'Distribution of Testing Error Magnitudes',...
    'using Maximum Likelihood Model Assignment'})
xlim([0,0.05])
drawnow

%% Evaluate model using decaying error norm

alpha = 2;
N_ahead = 10;


Errors_alpha = zeros(N_test-N_ahead,1);
for j = 1:N_test-N_ahead
    x = X_test(:,j);
    u = U_test(:,j);
    z = [x;u;1];
    
    E_alpha = 0;
    for n = 1:N_ahead
        [ y_hat, ~ ] = MLE_KernelModel_Eval( z, kernel, KerModels,...
        TrainData );
        
        % determine the error
        x = X_test(1:2,j+n);
        error = y_hat - x;
        
        % sum the alpha error norm
        E_alpha = E_alpha + sqrt(sum(error.^2))/alpha^(n-1);
        
        u = U_test(:,j+n);
        z = [y_hat;u;1]; 
    end
    
    Errors_alpha(j) = E_alpha;
    
end

E_alpha_avg = mean(Errors_alpha)

%% Generate new data using model
N_bases = TestDataFile.N_bases;

N_generate = 2000;
X_IC = [-0.49;0.43];
U_gen = zeros(N_bases,N_generate);

X_gen = zeros(2,N_generate);
X_gen(:,1) = X_IC;

MdlCounts_gen = zeros(1,N_Models);
MdlIndices_gen = ones(N_Models,N_generate);

for n = 2:N_generate
    x = X_gen(:,n-1);
    u = U_gen(:,n-1);
    z = [x;u;1];
    
    [ y_hat, i_star ] = MLE_KernelModel_Eval( z, kernel, KerModels,...
        TrainData );
    
    % MLE model estimate
    X_gen(:,n) = y_hat;
    
    MdlCounts_gen(i_star) = MdlCounts_gen(i_star) + 1;
    MdlIndices_gen(i_star, MdlCounts_gen(i_star)) = n-1;
    
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