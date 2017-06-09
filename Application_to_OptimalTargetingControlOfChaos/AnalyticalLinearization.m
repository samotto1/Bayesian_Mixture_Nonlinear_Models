function [A, B, y_nom] = AnalyticalLinearization(x, u, T_interval)

N_bases = length(u);

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

% Duffing ODE with additional forcing input
ode_fun = @(x,t,forcing) [  x(2);
              	gamma*cos(omega*t) - delta*x(2) - alpha*x(1) - ...
                beta*x(1).^3 + forcing];
            
J_fun = @(x) [  0, 1;
                -alpha-3*beta*x(1)^2, -delta ];
            
C_fun = @(x) [0; 1];
            
% generate trajectory
ops = odeset('RelTol', 1e-6);
f = @(t) bi(t,1:N_bases) * u;
[t_vec,X] = ode45(@(t,x) ode_fun(x,t,f(t)), T_interval, x, ops);
y_nom = X(end,:)';
x_traj = @(t) interp1(t_vec, X, t, 'pchip')';


% form fundamental matrix
[~,Psi_1] = ode45(@(t,y) J_fun(x_traj(t))*y, t_vec, [1;0], ops);
[~,Psi_2] = ode45(@(t,y) J_fun(x_traj(t))*y, t_vec, [0;1], ops);
Psi_traj = @(t) reshape(interp1(t_vec, [Psi_1, Psi_2], t, 'pchip')',2,2);
PsiInv_traj = @(t) pinv(Psi_traj(t));

A = Psi_traj(T_interval(2));

B = zeros(2,N_bases);
for j=1:N_bases
    B(:,j) = A*integral(@(t) PsiInv_traj(t)*C_fun(x_traj(t))*bi(t,j),...
        T_interval(1), T_interval(2), 'ArrayValued', true);
end
end