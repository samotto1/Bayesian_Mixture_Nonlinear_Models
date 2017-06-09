function [ y_hat ] = ith_KernelModel_Eval( z, i_star, kernel, KerModels, TrainData )
% Evaluate mixture of kernel models at given point
%   y_hat is the approximation given z=[x;u;1]
%   i_star is the model used to make the estimate

X_train = TrainData.X;
U_train = TrainData.U;

MdlCounts = KerModels.MdlCounts;
MdlIndices = KerModels.MdlIndices;
Mu_x = KerModels.Mu_x;
Mu_y = KerModels.Mu_y;
M_matrices = KerModels.M_matrices;

x_dim = size(X_train,1);

N_cluster = MdlCounts(i_star);
ker_inds = MdlIndices(i_star, 1:N_cluster);

% Evaluate kernel at point
K_z = zeros(N_cluster,1);
for l = 1:N_cluster
	K_z(l) = kernel([X_train(:,ker_inds(l));U_train(:,ker_inds(l));1],z);
end

% Evaluate kernel model
y_hat = M_matrices(:,ker_inds,i_star)*K_z + Mu_y(:,i_star);

end

