function resStruct = modifiedAndersonMooreSmoother(D1, D2, A, Z_tilde, Finv, K, a_t_t, P_t_t)

% check and extract dimensions
[dimObs, dimState, ~] = checkDimsModifiedSSM(D1, D2, A);
assert(size(Z_tilde,2) == dimObs)
nObs = size(Z_tilde,1);


D_tilde = (D1*A +D2);


% intialize struct for the results
resStruct = struct();
resStruct.a_t_T = nan(nObs, dimState);
resStruct.P_t_T = nan(dimState, dimState, nObs);

resStruct.a_t_T(nObs, :)     = a_t_t(nObs,:);
resStruct.P_t_T(:, :, nObs)  = P_t_t(:,:, nObs);

for iObs = nObs-1:-1:1
    a_filtered = a_t_t(iObs,:)';
    P_filtered = P_t_t(:,:, iObs);
    
    a_filtered_tp1    = a_t_t(iObs+1,:)';
    P_filtered_tp1    = P_t_t(:,:, iObs+1);
    Finv_tp1          = Finv(:,:, iObs+1);
    K_tp1             = K(:,:, iObs+1);
    Z_tilde_tp1       = Z_tilde(iObs+1,:)';
    
    % one-step ahead smoother
    xx = P_filtered * D_tilde';
    a_t_tp1 = a_filtered + xx * Finv_tp1 * Z_tilde_tp1;
    U_t_tp1 = P_filtered - xx * Finv_tp1 * xx';
    
    % components for J
    P_filtered_tp1_Inv = eye(size(P_filtered_tp1)) / P_filtered_tp1;
    
    xx = A * P_filtered - K_tp1 * D_tilde * P_filtered;
    P_t_tp1_given_tp1 = xx';
    
    J = P_t_tp1_given_tp1 * P_filtered_tp1_Inv;
    
    resStruct.a_t_T(iObs, :) = a_t_tp1 + ...
        J * (resStruct.a_t_T(iObs + 1, :)' - a_filtered_tp1);
    resStruct.P_t_T(:,:,iObs) = U_t_tp1 + ...
        J * (resStruct.P_t_T(:,:,iObs + 1) - P_filtered_tp1) * J';
    
    
end

end
