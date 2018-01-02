function resStruct = modifiedDeJongKohnAnsleySmoother(D1, D2, A, Z_tilde, Finv, K, a_t_t, P_t_t)

% check and extract dimensions
[dimObs, dimState, ~] = checkDimsModifiedSSM(D1, D2, A);
assert(size(Z_tilde,2) == dimObs)
nObs = size(Z_tilde,1);


D_tilde = (D1*A +D2);

% intialize struct for the results
resStruct = struct();
resStruct.a_t_T = nan(nObs, dimState);
resStruct.P_t_T = nan(dimState, dimState, nObs);

% initialize the smoother
r = zeros(dimState, 1);
N = zeros(dimState, dimState);

for iObs = nObs:-1:1
    Finv_t     = Finv(:,:, iObs);
    Z_tilde_t  = Z_tilde(iObs,:)';
    K_t        = K(:,:, iObs);
    a_filtered = a_t_t(iObs,:)';
    P_filtered = P_t_t(:,:, iObs);
    
    
    resStruct.a_t_T(iObs,:)   = a_filtered + P_filtered * r;
    resStruct.P_t_T(:,:,iObs) = P_filtered - P_filtered * N * P_filtered;
    
    if iObs == nObs
        L = 0;
    else
        L = A - K_t * D_tilde;
    end
    r = D_tilde' * Finv_t * Z_tilde_t + L' * r;
    N = D_tilde' * Finv_t * D_tilde + L' * N * L;
    
end

end
