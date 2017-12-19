function resStruct = modifiedSmoother2(Z, D1, D2, A, C, R, Z_tilde, Finv, K)


% check dimensions
assert(isequal(size(D1), size(D2)))
assert(size(A,1) == size(A,2))
assert(size(C,2) == size(R,2))

[nObs, dimObs] = size(Z);
dimState = size(A,1);
dimDisturbance = size(R,2);

D_tilde = (D1*A +D2);
D1CR    = (D1 * C + R);

resStruct = struct();
resStruct.a_t_T = nan(nObs, dimState);

% smooth disturbances
u_t_T = nan(nObs, dimDisturbance);

% initialize the smoother
r = zeros(dimState, 1);

%backward recursion
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

% forward recursion
for iObs = 1:nObs
    a_t_T = A * a_t_T + C * u_t_T(iObs, :)';
    
    resStruct.a_t_T(iObs, :) = a_t_T;
    
end



end
