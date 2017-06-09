function [ A, B, y_nom, i_star ] = MLE_KernelModel_Linearization( z, ...
    kernel, Dz_kernel, KerModels, TrainData )
% Evaluate mixture of kernel models at given point
%   y ~ y_nom + [A,B,0]*([x;u;1] - z)
%   y_nom is the approximation given z=[x;u;1]
%   i_star is the model used to make the estimate

NormalDist = @(x, mu, SigInv, SigDet) ...
    1/( (2*pi)^(length(x)/2)*sqrt(SigDet) ) *...
    exp( -0.5*(x-mu)'*SigInv*(x-mu) );

X_train = TrainData.X;
U_train = TrainData.U;

MdlCounts = KerModels.MdlCounts;
MdlIndices = KerModels.MdlIndices;
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

N_Models = length(phi_Mdls);
x_dim = size(X_train,1);
u_dim = size(U_train,1);

x = z(1:x_dim);

% Evaluate likelihood of each model
W_Mdls = zeros(1,N_Models);
for i = 1:N_Models
    W_Mdls(i) = phi_Mdls(i) * NormalDist(x, Mu_x(:,i), ...
        SigInv_Mdls(:,:,i), SigDet_Mdls(i));
end
W_Mdls = W_Mdls/sum(W_Mdls);

% Assign to model based on max likelihood
[~, i_star] = max(W_Mdls);

N_cluster = MdlCounts(i_star);
ker_inds = MdlIndices(i_star, 1:N_cluster);

% Evaluate kernel at point
K_z = zeros(N_cluster,1);
Dz_K = zeros(N_cluster,x_dim+u_dim+1);
for l = 1:N_cluster
	K_z(l) = kernel([X_train(:,ker_inds(l));U_train(:,ker_inds(l));1],z);
    Dz_K(l,:) = Dz_kernel([X_train(:,ker_inds(l));...
        U_train(:,ker_inds(l));1],z);
end

% Evaluate kernel model
y_nom = M_matrices(:,ker_inds,i_star)*K_z + Mu_y(:,i_star);

% Evaluate linearization
AB_mat = M_matrices(:,ker_inds,i_star)*Dz_K;
A = AB_mat(:,1:x_dim);
B = AB_mat(:,x_dim+1:x_dim+u_dim);
end

