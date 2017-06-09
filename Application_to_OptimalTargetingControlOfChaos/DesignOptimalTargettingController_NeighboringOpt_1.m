clear; clc; close all;

%% Parameters

eps_u = 0.05;%1e-1; % maximum control action magnitude
eps_x = 0.1; % maximum control neighborhood of FP

% number of random starting (root) points for optimal trajectories
N_RootPts = 10;

% control action weighting
R_u = 1e-3*eye(5);

% distance error growth factor in optimal contol cost function
gamma_x = 1.1; 

% tolerance level for optimal control convergence
tol_u_opt = 1e-6;

% file name to save optimal controller design
fname_OptControl = 'DuffingModel_OptTargController_Neighboring.mat';

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

%% Solve Optimal Targeting Control Problem

% state weighting in cost function
Q_x = K_FP'*Gw_bi*K_FP + 0.7*eye(2);

% Choose random starting points for optimal trajectories
root_pt_inds = randperm(N_train-1, N_RootPts)';

% Optimal nominal trajectories
opt_traj_len = ones(N_RootPts,1);
X_opt = zeros(2,N_RootPts,N_train);
U_opt = zeros(u_dim,N_RootPts,N_train);
Kvec_opt = zeros(2*u_dim,N_RootPts,N_train);

% Initialialize optimal trajectories by finding a point along the
% trajectory in the training data where the control action size needed to
% stablize the fixed point is minimized

for n_root=1:N_RootPts
    % start point index
    idx = root_pt_inds(n_root);
    
    % locate minimum required control action
    u_mag = zeros(1,N_train-idx+1);
    for j = idx:min([idx + 10, N_train]) %N_train
        x = X_train(:,j);
        u_mag(j-idx+1) = (x-x_FP)'*Q_x*(x-x_FP);
    end
    [~,idx_min] = min(u_mag);
    
    % use training data to initialize optimal trajectory
    opt_traj_len(n_root) = idx_min;
    for j=1:idx_min;
        X_opt(:,n_root,j) = X_train(:,idx+j-1);
        U_opt(:,n_root,j) = U_train(:,idx+j-1);
    end
end

%
% optimize each trajectory by solving finite horizon optimal control
% problem. Prune end of trajectory if it is sufficiently close to the fixed
% point to be controlled by the point controller.
%

% define the Hamiltonian and its gradient in the optimality condition
H = @(x,u,lam) 1/2*u'*R_u*u + ...
    1/2*lam'*MLE_KernelModel_Eval( [x;u;1], kernel, KerModels, TrainData );
H_Grad_u = @(x,u,lam) u'*R_u + 1/2*lam'*ModelLinearization_B_Matrix( x, u,...
    kernel, Dz_kernel, KerModels, TrainData );

% define nonlinear constraint and gradient for fmincon()
nonlcon = @(u) deal(u'*Gw_bi*u - eps_u^2, [], 2*Gw_bi*u, []);

% loop over starting (root) points
for n_root = 1:N_RootPts
    
    % initialize trajectory
    T_horiz = opt_traj_len(n_root); % horizon
    x_traj = zeros(2,T_horiz);
    u_traj = zeros(u_dim,T_horiz-1);
    for j = 1:T_horiz-1
        x_traj(:,j) = X_opt(:,n_root,j);
        u_traj(:,j) = U_opt(:,n_root,j);
    end
    x_traj(:,T_horiz) = X_opt(:,n_root,T_horiz);
    
    % initialize adjoint variables
    lambda = zeros(2,T_horiz-1);
    
    % iterate until convergence
    Delta_u_max = 1e6;
    iter = 0;
    figure()
    while Delta_u_max > tol_u_opt
        
        % prune the end of the trajectory if multiple points lie in the
        % controllable neighborhood of the fixed point. Update the finite
        % time horizon
        j_clip = 0;
        DX_traj = x_traj - x_FP*ones(1,T_horiz);
        while DX_traj(:,T_horiz-j_clip-1)'*K_FP'*Gw_bi*K_FP*DX_traj(:,T_horiz-j_clip-1) ...
                <= eps_u^2 && DX_traj(:,T_horiz-j_clip-1)'*DX_traj(:,T_horiz-j_clip-1) ...
                <= eps_x^2
            j_clip = j_clip + 1;
        end
        T_horiz = T_horiz-j_clip; % adjust horizon
        x_traj = x_traj(:,1:T_horiz);
        u_traj = u_traj(:,1:T_horiz-1);
        lambda = lambda(:,1:T_horiz-1);
        
        % update adjoint variables
        lambda(:,T_horiz-1) = 2*Q_x*(x_traj(:,T_horiz) - x_FP);
        for j=T_horiz-1:-1:2
            A = MLE_KernelModel_Linearization( ...
    [x_traj(:,j);u_traj(:,j);1], kernel, Dz_kernel, KerModels, TrainData );

            lambda(:,j-1) = 2*gamma_x^(j-T_horiz)*Q_x*(x_traj(:,j)-x_FP) +...
                A'*lambda(:,j);
             
            mag = sqrt(lambda(:,j-1)'*lambda(:,j-1));
            if mag > 1e2;
                lambda(:,j-1) = sqrt(lambda(:,j)'*lambda(:,j)) *...
                    lambda(:,j-1)/mag;
            end
        end     
        
        % determine optimal control and update trajectory
        u_new = zeros(u_dim,T_horiz-1);
        for j=1:T_horiz-1
            % optimize control
            objfun = @(u) deal(H(x_traj(:,j),u,lambda(:,j)),...
                H_Grad_u(x_traj(:,j),u,lambda(:,j))');
            u_new(:,j) = fmincon(objfun, u_traj(:,j),...
                [],[],[],[],[],[],nonlcon, fminops);
            
            fprintf('\t *** updated %d of %d trajectory points \n',...
                j, T_horiz-1);
        end
          
        for j=1:T_horiz-1
            % apply action to step
            x_traj(:,j+1) = MLE_KernelModel_Eval( ...
                [x_traj(:,j);u_new(:,j);1], kernel, KerModels,...
                TrainData );
        end
        
        % update control
        Delta_u_max = max(sqrt(sum((u_new-u_traj).^2,1)))
        u_traj = u_new;
        iter = iter + 1
        
        % plot the trajectory
        plot(X_train(1,:), X_train(2,:), 'k.', 'MarkerSize', 6);
        hold on
        plot(x_FP(1), x_FP(2), 'go', 'LineWidth', 3 ,'MarkerSize', 6);
        plot(x_traj(1,:), x_traj(2,:), 'gx-', 'LineWidth', 2);
        plot(x_traj(1,1), x_traj(2,1), 'bo', 'LineWidth', 3 ,...
            'MarkerSize', 6);
        hold off
        title(sprintf('Root Point = %d, Path Length = %d, Iteration = %d',...
            n_root,T_horiz,iter))
        drawnow
    end
    
    % update nominal optimal trajectory
    opt_traj_len(n_root) = T_horiz;
    for j = 1:T_horiz-1
        X_opt(:,n_root,j) = x_traj(:,j);
        U_opt(:,n_root,j) = u_traj(:,j);
    end
    X_opt(:,n_root,T_horiz) = x_traj(:,T_horiz);
    
    %
    % Design neighboring optimal controller
    %
    P = Q_x;
    for j = T_horiz-1:-1:1
        % linearized dynamics
        [A, B, ~, ~] = ModelLinearization_B_Matrix( x, u,...
            kernel, Dz_kernel, KerModels, TrainData );
        
        % determine neighboring optimal control gain
        M = pinv(R_u + B'*P*B);
        K = M*B*P*A;
        Kvec_opt(:,n_root,j) = reshape(K,2*u_dim,1);
        
        % update quadratic value function
        P = gamma_x^(j-T_horiz)*Q_x + A'*(P - P*B'*M*B*P)*A;
    end
end

save(fname_OptControl, 'N_bases', 'Gw_bi', 'A_FP', 'B_FP', 'K_FP', ...
    'R_u', 'Q_x', 'gamma_x', 'N_RootPts', 'root_pt_inds', 'opt_traj_len', ...
    'X_opt', 'U_opt', 'Kvec_opt', 'eps_x', 'eps_u', 'x_FP');