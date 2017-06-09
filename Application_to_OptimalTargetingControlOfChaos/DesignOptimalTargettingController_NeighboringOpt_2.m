clear; clc; close all;

%% Parameters

eps_u = 0.01;%1e-1; % maximum control action magnitude
eps_x = 0.1; % maximum control neighborhood of FP

% number of random starting (root) points for optimal trajectories
Root_Len = 50;

% control action weighting
R_u = 1e-3*eye(5);

% distance error growth factor in optimal contol cost function
gamma_x = 1.1; 

% tolerance level for optimal control convergence
tol_u_opt = 1e-6;

% file name to save optimal controller design
fname_OptControl = 'DuffingModel_OptTargController_Neighboring_1.mat';

% Duffing data with no forcing
ZeroForcingData = load('DuffingTrainData_ZeroForcing.mat');

fminops = optimoptions(@fmincon, 'Algorithm', 'interior-point',...
    'Display', 'off', ...
    'SpecifyObjectiveGradient',true, 'SpecifyConstraintGradient',true);

%% Define Duffing ODE

% Duffing Parameters
alpha = -1;
beta = 1;
delta = 0.25;
gamma = 0.30;
omega = 1.0;

omega_0 = omega;
T = 2*pi/omega;

% number of Fourier bases to use for forcing
N_bases = 5;

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


%% Load the Data
fname_ModelFit = 'DuffingKernelModelsFit.mat';
load(fname_ModelFit); %'KerModels', 'TrainData', 'kernel', 'Dz_kernel'

%% Use Models to Locate Unstable Periodic Orbit
X_train = TrainData.X;
U_train = TrainData.U;
Y_train = TrainData.Y;

N_train = size(X_train,2);
u_dim = size(U_train,1);

% Locate initial guess based on data
D = Y_train - X_train;
[~,idx] = min(sqrt(sum(D.^2,1)));
x_guess = 1/2 * (X_train(:,idx) + Y_train(:,idx));
x_guess = [0.1290;0.1937];

% Solve for fixed point using nonlinear models
FP_fun = @(x) x - MLE_KernelModel_Eval( [x;zeros(u_dim,1);1], kernel,...
    KerModels, TrainData );
x_FP = fsolve(FP_fun, x_guess)
%x_FP = [0.1290;0.1937]

% Model linearization about fixed point
[A_FP, B_FP, ~, ~] = MLE_KernelModel_Linearization( ...
    [x_FP;zeros(u_dim,1);1], kernel, Dz_kernel, KerModels, TrainData );

[A_FP1, B_FP1, ~] = AnalyticalLinearization(x_FP, zeros(N_bases,1), [0,T]);

[V,D] = eig(A_FP);
v_u = V(:,2);
v_s = V(:,1);

figure()
p1 = plot(X_train(1,:), X_train(2,:), 'k.', 'MarkerSize', 6);
hold on
p2 = plot(x_FP(1), x_FP(2), 'go', 'LineWidth', 3 ,'MarkerSize', 6);
len = 0.5;
p3 = plot([x_FP(1)-len*v_u(1),x_FP(1)+len*v_u(1)], ...
    [x_FP(2)-len*v_u(2),x_FP(2)+len*v_u(2)], 'r-', 'LineWidth', 1.5);
p4 = plot([x_FP(1)-len*v_s(1),x_FP(1)+len*v_s(1)], ...
    [x_FP(2)-len*v_s(2),x_FP(2)+len*v_s(2)], 'b-', 'LineWidth', 1.5);
hold off
grid on
legend([p1,p2,p3,p4], {'Training Data', 'Fixed Point', ...
    sprintf('Unstable Eigenspace, \\lambda_u = %.3f',D(2,2)),...
    sprintf('Stable Eigenspace, \\lambda_s = %.3f',D(1,1))});
title({'Fixed Point and Eigenspaces',...
    'Derived using Linearization of the Nonlinear Model'})

%% Design Point Controller to Stabilize UPO

% find orthogonal compliment to stable eigenspace
w_s = v_u - (v_u'*v_s)*v_s;
w_s = w_s/sqrt(w_s'*w_s);

Gw_inv = pinv(Gw_bi);
%K_FP = Gw_inv*B_FP'*w_s * pinv(w_s'*B_FP*Gw_inv*B_FP'*w_s) * w_s'*A_FP
%K_FP = Gw_inv*B_FP' * pinv(B_FP*Gw_inv*B_FP') * A_FP
K_FP = dlqr(A_FP, B_FP, 0.1*eye(2), Gw_bi)

%% Test Point Controller on Duffing Equation

x_IC = [0;0];

N_sim = 500;

X_sim = zeros(2,N_sim);
X_sim(:,1) = x_IC;
U_sim = zeros(N_bases,N_sim-1);

ops = odeset('RelTol', 1e-6);
for n = 1:N_sim-1
    
    % location
    x = X_sim(:,n);
    
    % determine control action
    u = -K_FP*(x-x_FP);
    
    if u'*Gw_bi*u > eps_u^2 || (x-x_FP)'*(x-x_FP) > eps_x^2
        u = zeros(N_bases,1);
    end
    
    U_sim(:,n) = u;
    
    % simulate using model
    z = [x;u;1];
    [ y_hat, i_star ] = MLE_KernelModel_Eval( z, kernel, KerModels,...
        TrainData );
    
    % simulate Duffing equation
    f = @(t) bi(t,1:N_bases) * u;
    [~,Y] = ode45(@(t,x) ode_fun(x,t,f(t)), [0,T], x, ops);
    
    X_sim(:,n+1) = Y(end,:)';
    %omega = 0.0;
    %X_sim(:,n+1) = omega*y_hat + (1-omega)*Y(end,:)';
    %y_hat-Y(end,:)'
end

figure()
subplot(3,1,1)
plot(X_sim(1,:), 'b+', 'LineWidth', 1.5);
xlim([0,N_sim])
ylabel('x_1')
title('Poincare Map with OGY Point Control')
subplot(3,1,2)
plot(X_sim(2,:), 'b+', 'LineWidth', 1.5);
xlim([0,N_sim])
ylabel('x_2')
subplot(3,1,3)
plot(sqrt(diag(U_sim'*Gw_bi*U_sim)), 'b+', 'LineWidth', 1.5);
xlim([0,N_sim])
ylabel('u^T*G*u')
xlabel('Map Iteration')
drawnow

%% Solve targetting control problem

% state weighting in cost function
Q_x = K_FP'*Gw_bi*K_FP + 0.7*eye(2);

% locate nearest point in unforced Poincare map to the fixed point based on
% the state weighting
X_ZeroForce = ZeroForcingData.X_data;
N_ZeroForce = size(X_ZeroForce,2);
u_dist = zeros(1,N_ZeroForce);
for n = 1:N_ZeroForce
    x = X_ZeroForce(:,n);
    u_dist(n) = (x-x_FP)'*Q_x*(x-x_FP);
end
[~,idx] = min(u_dist);

% Construct root trajectory
Root_Len = min(Root_Len, idx);
X_RootTraj = zeros(2,Root_Len);
for j = 1:Root_Len
    X_RootTraj(:,j) = X_ZeroForce(:,idx-Root_Len+j);
end

% Find LQ gain matrices at each point on root trajectory
P_mats = zeros(2,2,Root_Len);
K_RootTraj = zeros(u_dim,2,Root_Len-1);
P_mats(:,:,Root_Len) = Q_x;
for j = Root_Len:-1:2
    
    % state at (j-1)
    x = X_RootTraj(:,j-1);
    
    % compute linearization at (j-1)
    [A, B, ~, ~] = MLE_KernelModel_Linearization( ...
        [x;zeros(u_dim,1);1], kernel, Dz_kernel, KerModels, TrainData );
    
    % value function quadratic form at point (j)
    P = P_mats(:,:,j);
    
    % compute gain matrices
    M = pinv(R_u + B'*P*B);
    K_RootTraj(:,:,j-1) = M*B'*P*A;
    
    % update value function quadratic form
    P_mats(:,:,j-1) = gamma_x^(j-1-Root_Len) * Q_x + ...
        A'*(P - P*B*M*B'*P)*A;
    
end


save(fname_OptControl, 'N_bases', 'Gw_bi', 'A_FP', 'B_FP', 'K_FP', ...
    'R_u', 'Q_x', 'gamma_x', 'eps_x', 'eps_u', 'x_FP', 'Root_Len', ...
    'X_RootTraj', 'K_RootTraj');