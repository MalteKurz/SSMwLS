nObs = 10000;


A = 0.5;
C = [1, 0.5];
R = [0.5, 1];
D1 = 1;
D2 = 0.5;

% initialize
dimDisturbance = size(R,2);
dimState       = size(A,1);
dimObs         = size(D1,1);
Z              = nan(nObs, 1);
X              = nan(nObs, 1);

a_0_0 = zeros(dimState, 1);
CC = C*C';
P_0_0 = reshape(inv(eye(dimState*dimState)-kron(A,A))*CC(:),dimState,dimState);
X(1,:) = mvnrnd(a_0_0, P_0_0);

% simulate
u = randn(nObs, dimDisturbance);

for iObs = 1:nObs
    X(iObs+1,:) = A * X(iObs,:)' + C * u(iObs,:)';
    Z(iObs) = D1 * X(iObs+1,:)' + D2 * X(iObs,:)' + R * u(iObs,:)';
    
end


% smooth Nimark (2015)
smoothState = smooth(A,C,D1,D2,R,Z');
smoothState = smoothState(:,2:end)';

% filter
[negLogLike, resStructFilter] = modifiedFilter(Z, D1, D2, A, C, R);

% smooth corrected
resStructSmoother1 = modifiedSmoother1(Z, D1, D2, A, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K, resStructFilter.a_t_t, resStructFilter.P_t_t);
resStructSmoother2 = modifiedSmoother2(Z, D1, D2, A, C, R, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K);
resStructSmoother3 = modifiedSmoother3(Z, D1, D2, A, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.U, resStructFilter.a_t_t, resStructFilter.P_t_t);

% augmented system and smoothing
A_bar = [A zeros(dimState, dimState); eye(dimState), zeros(dimState, dimState)];
C_bar = [C; zeros(dimState, dimDisturbance)];
D1_bar = [D1, D2];
D2_bar = zeros(dimObs, 2*dimState);

smoothStateAugmented = smooth(A_bar,C_bar,D1_bar,D2_bar,R,Z');
smoothStateAugmented = smoothStateAugmented(:,2:end)';





