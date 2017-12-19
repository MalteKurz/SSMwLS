function [negLogLike, resStruct] = modifiedFilter(Z, D1, D2, A, C, R)

% check dimensions
assert(isequal(size(D1), size(D2)))
assert(size(A,1) == size(A,2))
assert(size(C,2) == size(R,2))

[nObs, dimObs] = size(Z);
dimState = size(A,1);

D_tilde = (D1*A +D2);

resStruct         = struct();
resStruct.Z_tilde = nan(nObs, dimObs);
resStruct.a_t_t   = nan(nObs, dimState);
resStruct.P_t_t   = nan(dimState, dimState, nObs);
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
    
    resStruct.Z_tilde(iObs,:) = Z_tilde;
    resStruct.a_t_t(iObs,:)   = a_t_t;
    resStruct.P_t_t(:,:,iObs) = P_t_t;
    resStruct.Finv(:,:,iObs)  = Finv;
    resStruct.K(:,:,iObs)     = K;
    resStruct.U(:,:,iObs)     = U;
    
    negLogLike =  negLogLike + dimObs*log(2*pi)/2 + 0.5* (log(det(F)) + Z_tilde' * Finv * Z_tilde);
    
end



end

