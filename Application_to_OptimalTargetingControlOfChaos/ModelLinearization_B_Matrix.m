function [ B ] = ModelLinearization_B_Matrix( x,u, ...
    kernel, Dz_kernel, KerModels, TrainData )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[~, B, ~, ~] = MLE_KernelModel_Linearization( ...
    [x;u;1], kernel, Dz_kernel, KerModels, TrainData );

end

