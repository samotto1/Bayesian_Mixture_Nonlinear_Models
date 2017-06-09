clear; clc; close all

%% Parameters

% mixture of nonlinear models file
fname_ModelFit = 'DuffingKernelModelsFit.mat';
load(fname_ModelFit); 
% Contents:
%'KerModels', 'TrainData', 'kernel', 'Dz_kernel'

% optimal controller design file
fname_OptControl = 'DuffingModel_OptTargController_Bellman_2.mat';
load(fname_OptControl);
% Contents:
% 'N_bases', 'Gw_bi', 'A_FP', 'B_FP', 'K_FP', ...
% 'R_u', 'Q_x', 'gamma_ValFun', 'N_ValFunPts', ...
% 'X_pts_ValFun', 'MdlIdx_pts', 'ValFun_pts', 'u_opt_pts', 'ValFun',...
% 'X_pts_extrap', 'ValFun_extrap', 'eps_x', 'eps_u', 'x_FP'
u_dim = N_bases;

% Number of iterations to simulate
N_sim = 50;

% Initial condition
x_IC = [-1.5; -0.6] + [3; 1.4].*rand(2,1);

% Decide to use targeting
UseTargeting = 'y';

% Adjust control region
eps_u = eps_u;
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
            
%% Plot the optimal value function

Nx_plt = 100;
Ny_plt = 100;
[X1_plt, X2_plt] = meshgrid(linspace(-1.5,1.5,Nx_plt),...
    linspace(-0.6,0.8,Ny_plt));

% plot the value function
ValFun_plt = zeros(Nx_plt, Ny_plt);
for i=1:Ny_plt
	for j=1:Nx_plt
        ValFun_plt(i,j) = ValFun([X1_plt(i,j); X2_plt(i,j)]);
	end
end

figure()
contourf(X1_plt, X2_plt, ValFun_plt, 15)
hold on
plot(TrainData.X(1,:), TrainData.X(2,:), 'k.', 'MarkerSize', 6)
plot(x_FP(1), x_FP(2), 'go', 'LineWidth', 3 ,'MarkerSize', 6);
hold off
colorbar
xlabel('x_1')
ylabel('x_2')
title('Optimal Fitted Value Function')
drawnow
            
%% Simulate controlled dynamics

fminops = optimoptions(@fmincon, 'Algorithm', 'interior-point',...
    'Display', 'off', 'SpecifyConstraintGradient',true);
nonlcon = @(u) deal(u'*Gw_bi*u - eps_u^2, [], 2*Gw_bi*u, []);

ode45ops = odeset('RelTol', 1e-6);

X_traj = zeros(2,N_sim);
U_traj = zeros(u_dim,N_sim-1);
ValFun_traj = zeros(1,N_sim);

X_traj(:,1) = x_IC;
ValFun_traj(1) = ValFun(x_IC);
for n = 1:N_sim-1
    
    x = X_traj(:,n);
    
    % determine if the point is sufficiently close to the fixed point to
    % use OGY
    if (x-x_FP)'*K_FP'*Gw_bi*K_FP*(x-x_FP) < eps_u^2 && ...
            (x-x_FP)'*(x-x_FP) < eps_x^2
        
        u = -K_FP*(x-x_FP);
        
    elseif strcmpi(UseTargeting, 'y') % use optimal targeting control
    
        % determine which model the current point belongs to
        [~, i_star] = MLE_KernelModel_Eval( [x;zeros(u_dim,1);1], ...
            kernel, KerModels, TrainData );

        % determine optimal control to minimize value function
        objfun = @(u) u'*R_u*u + gamma_ValFun * ...
            ValFun(ith_KernelModel_Eval( [x;u;1], i_star, kernel,...
            KerModels, TrainData ));
        u = fmincon(objfun, zeros(u_dim,1), [],[],[],[],[],[], ...
            nonlcon, fminops);
        
    else 
        u = zeros(u_dim,1);
    end
    
    U_traj(:,n) = u;
    
    % Simulate the dynamics
    f = @(t) bi(t,1:N_bases) * u;
    [~,Y] = ode45(@(t,x) ode_fun(x,t,f(t)), [0,T], x, ode45ops);
    
    X_traj(:,n+1) = Y(end,:)';
    ValFun_traj(n+1) = ValFun(X_traj(:,n+1));
    
    fprintf('Completed step %d of %d \n',n,N_sim)
end

%% Plot results of simulation

figure()
subplot(3,1,1)
plot(X_traj(1,:), 'b+', 'LineWidth', 1.5);
xlim([0,N_sim])
ylabel('x_1')
title('Poincare Map of Controlled Trajectory')
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
plot(ValFun_traj, 'b+-', 'LineWidth', 1.5);
grid on
xlabel('Map Iteration')
ylabel('Value Function')
title('Optimal Value Function along Controlled Trajectory')

figure()
contourf(X1_plt, X2_plt, ValFun_plt, 15)
hold on
plot(TrainData.X(1,:), TrainData.X(2,:), 'k.', 'MarkerSize', 6)
plot(x_FP(1), x_FP(2), 'go', 'LineWidth', 3 ,'MarkerSize', 6);
plot(X_traj(1,:), X_traj(2,:), 'g+-', 'LineWidth', 1.5);
hold off
colorbar
xlabel('x_1')
ylabel('x_2')
title('Trajectory with Optimal Targeting Control')
drawnow