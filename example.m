% One dimensional state
nObs = 1000;


A = 0.5;
C = [1, 0];
R = [0, 1];
D1 = 1;
D2 = 0.5;

% Three-dim state, two-dim observations, five-dim disturbances
A = diag([0.8,0.2,0.1]);
C = [diag([1, 0.9, 1.4]), zeros(3,2)];
R = [zeros(2,3), diag([0.8, 1.1])];
D1 = [1, 0.2, 0.1;...
    0.7, 0.9, 0.2];
D2 = [0.5, 0.1, 0.05;...
    0.9,0.05, 0.2];



% initialize
dimDisturbance = size(R,2);
dimState       = size(A,1);
dimObs         = size(D1,1);
Z              = nan(nObs, dimObs);
X              = nan(nObs, dimState);

a_0_0 = zeros(dimState, 1);
CC = C*C';
P_0_0 = reshape(inv(eye(dimState*dimState)-kron(A,A))*CC(:),dimState,dimState);
X(1,:) = mvnrnd(a_0_0, P_0_0);

% simulate
u = randn(nObs, dimDisturbance);

for iObs = 1:nObs
    X(iObs+1, :) = A * X(iObs,:)' + C * u(iObs,:)';
    Z(iObs, :)   = D1 * X(iObs+1,:)' + D2 * X(iObs,:)' + R * u(iObs,:)';
    
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
smoothStateAugmented = smoothStateAugmented(:,:)';





