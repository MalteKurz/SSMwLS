function [negLogLike, resStruct] = modifiedFilter(Z, D1, D2, A, C, R)

% check and extract dimensions
[dimObs, dimState, ~] = checkDimsModifiedSSM(D1, D2, A, C, R);
assert(size(Z,2) == dimObs)
nObs = size(Z,1);


D_tilde = (D1*A +D2);

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
a_t_t = zeros(dimState, 1);
CC=C*C';
P_t_t=reshape(inv(eye(dimState*dimState)-kron(A,A))*CC(:), dimState, dimState);

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

