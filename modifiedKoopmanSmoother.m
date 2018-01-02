function resStruct = modifiedKoopmanSmoother(D1, D2, A, C, R, Z_tilde, Finv, K)

% check and extract dimensions
[dimObs, dimState, dimDisturbance] = checkDimsModifiedSSM(D1, D2, A, C, R);
assert(size(Z_tilde,2) == dimObs)
nObs = size(Z_tilde,1);


D_tilde = (D1*A +D2);
D1CR    = (D1 * C + R);

% intialize struct for the results
resStruct = struct();
resStruct.a_t_T = nan(nObs, dimState);

% smooth disturbances
u_t_T = nan(nObs, dimDisturbance);

% initialize the smoother
r = zeros(dimState, 1);

% disturbance smoother (backward recursion)
for iObs = nObs:-1:1
    Finv_t     = Finv(:,:, iObs);
    Z_tilde_t  = Z_tilde(iObs,:)';
    K_t        = K(:,:, iObs);
    
    M = C - K_t *D1CR;
    u_t_T(iObs, :) = D1CR' * Finv_t * Z_tilde_t + M' * r;
    
    if iObs == nObs
        L = 0;
    else
        L = A - K_t * D_tilde;
    end
    r = D_tilde' * Finv_t * Z_tilde_t + L' * r;
    
end

% initialization
a_0_0 = zeros(dimState, 1);
CC=C*C';
P_0_0=reshape(inv(eye(dimState*dimState)-kron(A,A))*CC(:),dimState,dimState);
a_t_T = a_0_0 + P_0_0 * r;

% state smoother (forward recursion)
for iObs = 1:nObs
    a_t_T = A * a_t_T + C * u_t_T(iObs, :)';
    
    resStruct.a_t_T(iObs, :) = a_t_T;
    
end



end
