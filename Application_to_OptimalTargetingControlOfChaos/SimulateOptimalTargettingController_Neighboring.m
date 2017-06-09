clear; clc; close all

%% Parameters

% mixture of nonlinear models file
fname_ModelFit = 'DuffingKernelModelsFit.mat';
load(fname_ModelFit); 
% Contents:
%'KerModels', 'TrainData', 'kernel', 'Dz_kernel'

% optimal controller design file
fname_OptControl = 'DuffingModel_OptTargController_Neighboring_1.mat';
load(fname_OptControl);
% Contents:
% 'N_bases', 'Gw_bi', 'A_FP', 'B_FP', 'K_FP', ...
% 'R_u', 'Q_x', 'gamma_x', 'eps_x', 'eps_u', 'x_FP', 'Root_Len', ...
% 'X_RootTraj', 'K_RootTraj'
u_dim = N_bases;

% Number of iterations to simulate
N_sim = 50;

% Initial condition
x_IC = [-1.5; -0.6] + [3; 1.4].*rand(2,1);

% Decide to use targeting
UseTargeting = 'y';

% Adjust control region
eps_u = 0.01; %eps_u;
eps_x = eps_x;


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
            
%% Plot the root points used for targeting control 

figure()
plot(TrainData.X(1,:), TrainData.X(2,:), 'k.', 'MarkerSize', 6)
hold on
plot(X_RootTraj(1,:), X_RootTraj(2,:), 'bx', 'MarkerSize', 6, ...
    'LineWidth', 1.5)
plot(x_FP(1), x_FP(2), 'go', 'LineWidth', 3 ,'MarkerSize', 6);
hold off
xlabel('x_1')
ylabel('x_2')
title('Optimal Fitted Value Function')
grid on
drawnow
            
%% Simulate controlled dynamics

ode45ops = odeset('RelTol', 1e-6);

X_traj = zeros(2,N_sim);
U_traj = zeros(u_dim,N_sim-1);

X_traj(:,1) = x_IC;
for n = 1:N_sim-1
    
    x = X_traj(:,n);
    
    % determine if the point is sufficiently close to the fixed point to
    % use OGY
    if (x-x_FP)'*K_FP'*Gw_bi*K_FP*(x-x_FP) < eps_u^2 && ...
            (x-x_FP)'*(x-x_FP) < eps_x^2
        
        u = -K_FP*(x-x_FP);
        
    elseif strcmpi(UseTargeting, 'y') % use optimal targeting control
    
        % locate first point on root trajectory which can be controlled
        can_control = 0;
        j = 0;
        while can_control == 0 && j < Root_Len-1
            x_root = X_RootTraj(:,Root_Len-1-j);
            K_root = K_RootTraj(:,:,Root_Len-1-j);
            u_tmp = -K_root*(x - x_root);
            
            if u_tmp'*Gw_bi*u_tmp <= eps_u^2
                u = u_tmp;
                can_control = 1;
            else
                j = j+1;
            end
        end
        
        if j == Root_Len-1
            u = zeros(u_dim,1);
        end
        
    else 
        u = zeros(u_dim,1);
    end
    
    U_traj(:,n) = u;
    
    % Simulate the dynamics
    f = @(t) bi(t,1:N_bases) * u;
    [~,Y] = ode45(@(t,x) ode_fun(x,t,f(t)), [0,T], x, ode45ops);
    
    X_traj(:,n+1) = Y(end,:)';
    
    %fprintf('Completed step %d of %d \n',n,N_sim)
end

%% Plot results of simulation

figure()
subplot(3,1,1)
plot(X_traj(1,:), 'b+', 'LineWidth', 1.5);
xlim([0,N_sim])
ylabel('x_1')
title('Poincare Map with OGY Point Control')
subplot(3,1,2)
plot(X_traj(2,:), 'b+', 'LineWidth', 1.5);
xlim([0,N_sim])
ylabel('x_2')
subplot(3,1,3)
plot(sqrt(diag(U_traj'*Gw_bi*U_traj)), 'b+', 'LineWidth', 1.5);
xlim([0,N_sim])
ylabel('u^T*G*u')
xlabel('Map Iteration')

figure()
plot(TrainData.X(1,:), TrainData.X(2,:), 'k.', 'MarkerSize', 6)
hold on
plot(x_FP(1), x_FP(2), 'go', 'LineWidth', 3 ,'MarkerSize', 6);
plot(X_traj(1,:), X_traj(2,:), 'g+-', 'LineWidth', 1.5);
plot(X_RootTraj(1,:), X_RootTraj(2,:), 'bx', 'MarkerSize', 6,...
    'LineWidth', 1.5)
hold off
xlabel('x_1')
ylabel('x_2')
title('Trajectory with Optimal Targeting Control')
grid on
drawnow