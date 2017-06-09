clear; clc; close all;

%% Parameters

eps_u = 0.05;%1e-1; % maximum control action magnitude
eps_x = 0.1; % maximum control neighborhood of FP

% number of Fourier bases to use for forcing
N_bases = 5;

% number of random points for value function interpolaton
N_ValFunPts = 300;

% control weighting in value function
R_u = 0.1 * eye(N_bases);

% value function tolerance
tol_ValFun = 1e-6;

% value function discount factor
gamma_ValFun = 1.0;

fminops = optimoptions(@fmincon, 'Algorithm', 'interior-point',...
    'Display', 'off', 'SpecifyConstraintGradient',true);

% file name to save optimal controller design
fname_OptControl = 'DuffingModel_OptTargController_Bellman_1.mat';

%% Define Duffing ODE

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

%[A_FP1, B_FP1, ~] = AnalyticalLinearization(x_FP, zeros(N_bases,1), [0,T]);

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

N_sim = 2000;

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
    
%     % simulate using model
%     z = [x;u;1];
%     [ y_hat, i_star ] = MLE_KernelModel_Eval( z, kernel, KerModels,...
%         TrainData );
    
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

%% Solve Optimal Targeting Control Problem using Fitted Value Iteration

% state weighting in value function
Q_x = K_FP'*Gw_bi*K_FP + 0.7*eye(2);

% initialization used for interpolation refinement 
% (by simply re-running section)
ValFun_Init = @(x) 1/2*(x-x_FP)'*Q_x*(x-x_FP); % ValFun(x)
N_ValFunPts = N_ValFunPts; % 2*N_ValFunPts

% Choose random starting points for optimal trajectories
ValFun_pt_inds = randperm(N_train, N_ValFunPts-1);
X_pts_ValFun = [x_FP, X_train(:,ValFun_pt_inds)];

% determine the maximum likelihood model at each point
MdlIdx_pts = zeros(1,N_ValFunPts);
for j=1:N_ValFunPts
    x = X_pts_ValFun(:,j);
    [~, i_star] = MLE_KernelModel_Eval( [x;zeros(u_dim,1);1], kernel,...
    KerModels, TrainData );
    MdlIdx_pts(j) = i_star;
end

% initialize value function at points 
ValFun_pts = zeros(1,N_ValFunPts);
for j=1:N_ValFunPts
    ValFun_pts(j) = ValFun_Init(X_pts_ValFun(:,j));
end

X_pts_extrap = [    -5, -5, 5,  5;
                    -5, 5,  -5, 5   ];
ValFun_extrap = max(ValFun_pts) * ones(1,4);

% initialize optimal control action at points
u_opt_pts = zeros(N_bases, N_ValFunPts);

% form grid for contour plotting
Nx_plt = 50;
Ny_plt = 50;
[X1_plt, X2_plt] = meshgrid(linspace(-1.5,1.5,Nx_plt),...
    linspace(-0.6,0.8,Ny_plt));

% fitted value iteration loop
MaxValFunChange = 1e6;
figure()
iter = 0;
while MaxValFunChange > tol_ValFun && iter < 100
    
    % update value function
    MaxValFunChange = 0;
    rand_pt_cycle = randperm(N_ValFunPts); % random point cycling
    for jj=1:N_ValFunPts
        j = rand_pt_cycle(jj);
        
        % construct value function interpolant
        ValFun_extrap = max(ValFun_pts) * ones(1,4);
        ValFun = @(x) griddata([X_pts_ValFun(1,:), X_pts_extrap(1,:)],...
            [X_pts_ValFun(2,:), X_pts_extrap(2,:)],...
            [ValFun_pts, ValFun_extrap], x(1), x(2), 'natural');
        
        x = X_pts_ValFun(:,j);
        u_guess = u_opt_pts(:,j);
        i_star = MdlIdx_pts(j);
        
        % find optimal control action
        nonlcon = @(u) deal(u'*Gw_bi*u - eps_u^2, [], 2*Gw_bi*u, []);
        objfun = @(u) u'*R_u*u + gamma_ValFun * ...
            ValFun(ith_KernelModel_Eval( ...
            [x;u;1], i_star, kernel, KerModels, TrainData ));
        u_star = fmincon(objfun, u_guess,...
                    [],[],[],[],[],[],nonlcon, fminops);
        u_opt_pts(:,j) = u_star;
        
        % update value function
        ValFun_pt_new = 1/2*(x-x_FP)'*Q_x*(x-x_FP) + ...
            1/2*u_star'*R_u*u_star + gamma_ValFun * ...
            ValFun(ith_KernelModel_Eval( ...
            [x;u_star;1], i_star, kernel, KerModels, TrainData ));
        
        if abs(ValFun_pt_new - ValFun_pts(j)) > MaxValFunChange
            MaxValFunChange = abs(ValFun_pt_new - ValFun_pts(j));
        end
        
        fprintf('Updated %d of %d : DeltaV = %.3e \n', jj,...
            N_ValFunPts,abs(ValFun_pt_new - ValFun_pts(j)))
        
        ValFun_pts(j) = ValFun_pt_new;
    end
    
    iter = iter + 1
    
    % plot the value function
    ValFun_plt = zeros(Nx_plt, Ny_plt);
    for i=1:Ny_plt
        for j=1:Nx_plt
            ValFun_plt(i,j) = ValFun([X1_plt(i,j); X2_plt(i,j)]);
        end
    end
    contourf(X1_plt, X2_plt, ValFun_plt)
    hold on
    plot(X_pts_ValFun(1,:), X_pts_ValFun(2,:), 'kx', 'LineWidth', 1.5);
    hold off
    xlabel('x_1')
    ylabel('x_2')
    title(sprintf('Value Function on Iteration = %d', iter))
    colorbar
    drawnow
end

save(fname_OptControl, 'N_bases', 'Gw_bi', 'A_FP', 'B_FP', 'K_FP', ...
    'R_u', 'Q_x', 'gamma_ValFun', 'N_ValFunPts', 'ValFun_pt_inds', ...
    'X_pts_ValFun', 'MdlIdx_pts', 'ValFun_pts', 'u_opt_pts', 'ValFun',...
    'X_pts_extrap', 'ValFun_extrap', 'eps_x', 'eps_u', 'x_FP');