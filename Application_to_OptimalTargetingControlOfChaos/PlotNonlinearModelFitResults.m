clear; clc; close all;

fname_TestData = 'DuffingTestData.mat';

fname_UnForced = 'DuffingTrainData_ZeroForcing.mat';

fname_ModelFit = 'DuffingKernelModelsFit.mat';
load(fname_ModelFit);

%% load the data

TestDataFile = load(fname_TestData);

N_test = TestDataFile.N_samples;
X_test = TestDataFile.X_data;
U_test = TestDataFile.U_data;
Y_test = TestDataFile.Y_data;

% model parameters
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

%% Evaluate Models on Testing Data 

% Initialize point allocations based on maximum likelihood
N_Models = length(phi_Mdls);
MdlCounts_MLE = zeros(1,N_Models);
MdlIndices_MLE = ones(N_Models,N_test);

% Initialize estimates
Y_hat_MLE = zeros(2,N_test); % using maximum likelihood model

% Assign points to models and construct estimates
W_Mdls = zeros(1,N_Models);
N_train = size(TrainData.X,2);
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
%X_IC = [-0.49;0.43];
%X_IC = [-1.5;-0.6] + [3.0;1.4].*rand(2,1)
X_IC = X_test(:,randi(N_test))
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

%% Look at cross-correlation

% load un-forced data
UnForcedData = load(fname_UnForced);
X_data = UnForcedData.X_data;

% find cross-correlations
N_lag = 30;

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
title('n-back Autocorrelations in Data')
ylabel('E[x_1(j)x_1(j-n)]')
legend([p1,p2], {'Duffing Data', 'Generative Model'})

subplot(3,1,2)
stairs(lags(N_lag+1:end), corr_xy_data(N_lag+1:end), 'k-', 'LineWidth', 1.5);
hold on
stairs(lags(N_lag+1:end), corr_xy_gen(N_lag+1:end), 'b-', 'LineWidth', 1.5);
hold off
grid on
ylabel('E[x_1(j)x_2(j-n)]')

subplot(3,1,3)
stairs(lags(N_lag+1:end), corr_yy_data(N_lag+1:end), 'k-', 'LineWidth', 1.5);
hold on
stairs(lags(N_lag+1:end), corr_yy_gen(N_lag+1:end), 'b-', 'LineWidth', 1.5);
hold off
grid on
ylabel('E[x_2(j)x_2(j-n)]')
xlabel('Lag, n')