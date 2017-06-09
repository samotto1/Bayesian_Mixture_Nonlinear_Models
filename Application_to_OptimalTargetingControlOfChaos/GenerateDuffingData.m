clear; clc; close all

fname = 'DuffingTrainData.mat';
%fname = 'DuffingTestData.mat';

% number of Fourier bases to use for forcing
N_bases = 5;
% size of forcing perturbations
pert_u = 1e-2;

% Number of samples
N_samples = 5000;

% Duffing Parameters
alpha = -1;
beta = 1;
delta = 0.25;
gamma = 0.30;
omega = 1.0;

omega_0 = omega;
T = 2*pi/omega;

% initial condition
x_IC = [0;0];
%x_IC = [-0.5;-0.5] + 1.0*rand(2,1);

%% Constuct the Duffing ODE

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
            
%% Generate Data

ops = odeset('RelTol', 1e-6);

X_data = zeros(2,N_samples);
U_data = zeros(N_bases,N_samples);
Y_data = zeros(2,N_samples);
for j = 1:N_samples
    % random time-dependent forcing
    u = pert_u*(2*rand(N_bases,1) - 1);
    f = @(t) bi(t,1:N_bases) * u;
    
    X_data(:,j) = x_IC;
    U_data(:,j) = u;
    
    % run simulation
    [~,Y] = ode45(@(t,x) ode_fun(x,t,f(t)), [0,T], x_IC, ops);
    Y_data(:,j) = Y(end,:)';
    
    % update
    x_IC = Y(end,:)';
end

figure()
f1_p1 = plot(Y_data(1,:), Y_data(2,:), 'k.');
title('Perturbed Duffing Poincare Map')
grid on
drawnow

%% Save the data
save(fname, 'N_samples', 'X_data', 'U_data', 'Y_data', 'N_bases',...
    'N_b_cos', 'N_b_sin', 'bi', 'Gw_bi');