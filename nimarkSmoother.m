function resStruct = nimarkSmoother(D1, D2, A, Finv, U, K, a_t_t, P_t_t, P_tp1_t)


% check dimensions
assert(isequal(size(D1), size(D2)))
assert(size(A,1) == size(A,2))

[nObs, dimState] = size(a_t_t);

D_tilde = (D1*A +D2);

resStruct = struct();
resStruct.a_t_T = nan(nObs, dimState);
resStruct.P_t_T = nan(dimState, dimState, nObs);

% initialize the matrices for the variance of the smoother
N = zeros(dimState, dimState);
M = zeros(dimState, dimState);

resStruct.a_t_T(nObs,:) = a_t_t(nObs,:);

for iObs = nObs-1:-1:1
    a_filtered = a_t_t(iObs,:)';
    P_filtered = P_t_t(:,:, iObs);
    Finv_t     = Finv(:,:, iObs);
    U_t        = U(:,:, iObs);
    K_t        = K(:,:, iObs);
    
    xx = P_tp1_t(:,:, iObs);
    P_tp1_t_inv = eye(size(xx)) / xx;
    
    J = P_filtered * A' * P_tp1_t_inv;
    
    resStruct.a_t_T(iObs,:)   = a_filtered + J * (resStruct.a_t_T(iObs + 1,:)' - A * a_filtered);
    xx = J * M * P_filtered;
    resStruct.P_t_T(:,:,iObs) = P_filtered + J * N * J' - xx - xx';
    

    L = A - K_t * D_tilde;
    
    KFK = U_t * Finv_t * U_t';
    
    N = KFK + J * N * J';
    M = K_t * D_tilde + J * M * L;
    
end

end
