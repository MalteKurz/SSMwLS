function [negLogLike, resStruct] = modifiedFilter(Z, D1, D2, A, C, R)
%MODIFIEDFILTER Nimark's (2015) modified Kalman filter for SSMwLS
% Purpose
%        The function computes Nimark's (2015) modified Kalman filter for
%        State Space Models with Lagged State (SSMwLS) in the measurement
%        equation.
%
%
% Usage
%        For SSMwLS
%           [negLogLike, resStruct] = modifiedFilter(Z, D1, D2, A, C, R)
%        For a classical SSM, i.e., without lagged state
%           [negLogLike, resStruct] = modifiedFilter(Z, D1, zeros(size(D1)), A, C, R)
%
%
% Model Equation
%       Measurement Equation
%           Z_t = D_1 X_t + D_2 X_t-1 + R u_t
%       State Equation
%           X_t = A X_t-1 + C u_t
%
% Inputs
%       Z  = (nObs x dimObs) vector of observables
%       D1 = (dimObs x dimState) matrix from the measurement equation
%       D2 = (dimObs x dimState) matrix from the measurement equation
%       A  = (dimState x dimState) matrix from the state equation
%       C  = (dimState x dimDisturbance) matrix from the state equation
%       R  = (dimObs x dimDisturbance) matrix from the measurement equation 
%
%
% Outputs
%      negLogLike = The negative log-likelihood
%      resStruct  = A structure containing
%                       Z_tilde Errors
%                       a_t_t   Filtered states
%                       P_t_t   Filtered variances
%                       P_t_tp1 One-step ahead predictors of the variances
%                       Finv    Second term of the Kalman gain
%                       K       Kalman gain
%                       U       First term of the Kalman gain
%
%
% References
%      Nimark, K. P. 2015. "A low dimensional Kalman filter for systems
%         with lagged states in the measurement equation". Economics
%         Letters 127: 10-13.
%
%
%
% Author: Malte Kurz


% check and extract dimensions
[dimObs, dimState] = checkDimsModifiedSSM(D1, D2, A, C, R);
assert(size(Z,2) == dimObs)
nObs = size(Z,1);


D_tilde = (D1*A +D2);
CC = C * C';

% intialize struct for the results
resStruct         = struct();
resStruct.Z_tilde = nan(nObs, dimObs);
resStruct.a_t_t   = nan(nObs, dimState);
resStruct.P_t_t   = nan(dimState, dimState, nObs);
resStruct.P_tp1_t = nan(dimState, dimState, nObs);
resStruct.Finv    = nan(dimObs, dimObs, nObs);
resStruct.K       = nan(dimState, dimObs, nObs);
resStruct.U       = nan(dimState, dimObs, nObs);

% initialize filter
[a_t_t, P_t_t] = initializeSSM(A, C, dimState);


negLogLike = 0;

for iObs = 1:nObs
    
    Z_tilde = Z(iObs, :)' - D_tilde*a_t_t;
    U = A * P_t_t * D_tilde' + CC * D1' + C * R';
    F = D_tilde * P_t_t * D_tilde' + (D1 * C + R) * (D1 * C + R)';
    
    Finv = eye(size(F)) / F;
    K = U * Finv;
    
    a_t_t = A * a_t_t + K * Z_tilde;
    P_t_t = A * P_t_t * A' + CC - K*F*K';
    
    P_tp1_t = A * P_t_t * A' + CC;
    
    resStruct.Z_tilde(iObs,:)   = Z_tilde;
    resStruct.a_t_t(iObs,:)     = a_t_t;
    resStruct.P_t_t(:,:,iObs)   = P_t_t;
    resStruct.P_tp1_t(:,:,iObs) = P_tp1_t;
    resStruct.Finv(:,:,iObs)    = Finv;
    resStruct.K(:,:,iObs)       = K;
    resStruct.U(:,:,iObs)       = U;
    
    negLogLike =  negLogLike + dimObs*log(2*pi)/2 + 0.5* (log(det(F)) + Z_tilde' * Finv * Z_tilde);
    
end



end

