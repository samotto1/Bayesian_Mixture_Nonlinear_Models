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
N_sim_max = 10000;

% Number of trials to average over at each max control size
N_trials = 10;

% Control action sizes to test
N_eps_u = 5;
eps_u_vec = logspace(log10(0.01), log10(0.05), N_eps_u);

% Decide to use targeting
UseTargeting = 'y';

% Adjust control region
eps_x = 0.2;

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

%% Loop over control action sizes and calulate expected time to control

fminops = optimoptions(@fmincon, 'Algorithm', 'interior-point',...
    'Display', 'off', 'SpecifyConstraintGradient',true);

ode45ops = odeset('RelTol', 1e-6);

ControlTimes = zeros(N_eps_u,N_trials);
for j = 1:N_eps_u
    eps_u = eps_u_vec(j);
    nonlcon = @(u) deal(u'*Gw_bi*u - eps_u^2, [], 2*Gw_bi*u, []);
    
    % loop over many random initial conditions
    for k = 1:N_trials
        % Choose random initial condition
        x_IC = [-1.5;-0.6] + [3.0;1.4].*rand(2,1);
        
        % simulate the dynamics until the point remains inside the
        % controllable region for several consecutive iterations
        n_iter = 0;
        interior_count = 0;
        x = x_IC;
        while n_iter <= N_sim_max && interior_count < 10
            % Determine control action
            if (x-x_FP)'*K_FP'*Gw_bi*K_FP*(x-x_FP) < eps_u^2 && ...
                (x-x_FP)'*(x-x_FP) < eps_x^2
        
                u = -K_FP*(x-x_FP);
                
                interior_count = interior_count + 1;
        
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
                
                interior_count = 0;
        
            else 
                u = zeros(u_dim,1);
                
                interior_count = 0;
            end
            
            % simulate the dynamics
            f = @(t) bi(t,1:N_bases) * u;
            [~,Y] = ode45(@(t,x) ode_fun(x,t,f(t)), [0,T], x, ode45ops);
    
            x = Y(end,:)';
            n_iter = n_iter + 1;
    
            fprintf('eps_u = %.3e, trial %d of %d, sim iter %d \n',...
                eps_u, k, N_trials, n_iter);
            
        end
        
        fprintf('\n eps_u = %.3e (%d of %d), trial %d of %d, Control Time = %d \n',...
            eps_u, j, N_eps_u, k, N_trials, n_iter - interior_count)
        ControlTimes(j,k) = n_iter - interior_count;
    end
end

%% Plot time to control

eps_u_mat = eps_u_vec' * ones(1,N_trials);

MeanControlTimes = mean(ControlTimes,2); 

loglogfit = polyfit(log(eps_u_vec(:)), log(MeanControlTimes(:)), 1);


figure()
plt_1 = loglog(eps_u_mat(:), ControlTimes(:), 'b+', 'LineWidth', 1.5);
hold on
plt_2 = loglog(eps_u_vec(:), MeanControlTimes(:), 'ko', 'LineWidth', 1.5);
plt_3 = loglog(eps_u_vec, exp(polyval(loglogfit, log(eps_u_vec))),...
    'k-', 'LineWidth', 1.5);
hold off
xlim([0.01,0.05]);
set(gca, 'XTick', linspace(0.01,0.05,5));
grid on
legend([plt_1, plt_2, plt_3], {'Simulated Trials', 'Mean Times to Control', ...
    sprintf('<\\tau> = (%.3e)\\epsilon_u^{%.3f}',exp(loglogfit(2)), loglogfit(1))})
xlabel('Max Control Size, \epsilon_u')
ylabel('Time to Control')
title('Time to Control vs Max Control Size')