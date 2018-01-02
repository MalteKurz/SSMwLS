nObs = 5000;

%% Example 1: One dimensional state

% A = 0.9;
% C = [1, 0];
% R = [0, 1];
% D1 = 1;
% D2 = 0.5;

%% Example 2: Three-dim state, two-dim observations, five-dim disturbances
A = diag([0.8,0.2,0.1]);
C = [diag([1, 0.9, 1.4]), zeros(3,2)];
R = [zeros(2,3), diag([0.8, 1.1])];
D1 = [1, 0.2, 0.1;...
    0.7, 0.9, 0.2];
D2 = [0.5, 0.1, 0.05;...
    0.9,0.05, 0.2];



%% initialize
dimDisturbance = size(R,2);
dimState       = size(A,1);
dimObs         = size(D1,1);
Z              = nan(nObs, dimObs);
X              = nan(nObs, dimState);

a_0_0 = zeros(dimState, 1);
CC = C*C';
P_0_0 = reshape(inv(eye(dimState*dimState)-kron(A,A))*CC(:),dimState,dimState);
X(1,:) = mvnrnd(a_0_0, P_0_0);

%% simulate
u = randn(nObs, dimDisturbance);

for iObs = 1:nObs
    X(iObs+1, :) = A * X(iObs,:)' + C * u(iObs,:)';
    Z(iObs, :)   = D1 * X(iObs+1,:)' + D2 * X(iObs,:)' + R * u(iObs,:)';
    
end

%% smooth

% filter
[negLogLike, resStructFilter] = modifiedFilter(Z, D1, D2, A, C, R);

% Implementation of the Nimark (2015) smoother
resStructNimarkSmoother = nimarkSmoother(D1, D2, A, ...
    resStructFilter.Finv, resStructFilter.U, resStructFilter.K, resStructFilter.a_t_t, resStructFilter.P_t_t, resStructFilter.P_tp1_t);

% smooth corrected
resStructSmoother1 = modifiedSmoother1(Z, D1, D2, A, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K, resStructFilter.a_t_t, resStructFilter.P_t_t);

resStructSmoother2 = modifiedSmoother2(Z, D1, D2, A, C, R, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K);

resStructSmoother3 = modifiedSmoother3(Z, D1, D2, A, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.U, resStructFilter.a_t_t, resStructFilter.P_t_t);

%% comparison
disp('Modified: Smoother 1 (Eq. (4.12)) vs. Smoother 2 (Eq. (4.16))')
max(max(resStructSmoother1.a_t_T - resStructSmoother2.a_t_T))
disp('Modified: Smoother 1 (Eq. (4.12)) vs. Smoother 3 (Eq. (4.3))')
max(max(resStructSmoother1.a_t_T - resStructSmoother3.a_t_T))

disp('Modified: Smoother 1 (Eq. (4.12)) vs. Nimark Smoother (Eq. (3.2))')
max(max(resStructSmoother1.a_t_T - resStructNimarkSmoother.a_t_T))



%% augmented system and smoothing
A_bar = [A zeros(dimState, dimState); eye(dimState), zeros(dimState, dimState)];
C_bar = [C; zeros(dimState, dimDisturbance)];
D1_bar = [D1, D2];
D2_bar = zeros(dimObs, 2*dimState);

% augmented system with MATLAB build-in
xx = C(1:dimState, 1:dimState);
C_tilde = [xx; zeros(dimState, dimState)];
R_tilde = R(1:dimObs, dimState+1:dimState+dimObs);

% synchronize the initial conditions
a_0_0 = zeros(dimState*2, 1);
CC = C_tilde*C_tilde';
P_0_0 = reshape(inv(eye(dimState*2*dimState*2)-kron(A_bar,A_bar))*CC(:),dimState*2,dimState*2);
a_1 = a_0_0;
P_1_1 = A_bar * P_0_0 * A_bar' + CC;

mdl = ssm(A_bar, C_tilde, D1_bar, R_tilde);
filteredStatesBuildIn = filter(mdl, Z);
smoothStatesBuildIn = smooth(mdl,Z);

% augmented system smoothing with corrected smoothers
[negLogLike, resStructFilterAugmented] = modifiedFilter(Z, D1_bar, D2_bar, A_bar, C_bar, R);

% Implementation of the Nimark (2015) smoother
resStructNimarkSmootherAugmented = nimarkSmoother(D1_bar, D1_bar, A_bar, ...
    resStructFilterAugmented.Finv, resStructFilterAugmented.U, resStructFilterAugmented.K, resStructFilterAugmented.a_t_t, resStructFilterAugmented.P_t_t, resStructFilterAugmented.P_tp1_t);

resStructSmoother1Augmented = modifiedSmoother1(Z, D1_bar, D2_bar, A_bar, ...
    resStructFilterAugmented.Z_tilde, resStructFilterAugmented.Finv, resStructFilterAugmented.K, resStructFilterAugmented.a_t_t, resStructFilterAugmented.P_t_t);

resStructSmoother2Augmented = modifiedSmoother2(Z, D1_bar, D2_bar, A_bar, C_bar, R, ...
    resStructFilterAugmented.Z_tilde, resStructFilterAugmented.Finv, resStructFilterAugmented.K);

resStructSmoother3Augmented = modifiedSmoother3(Z, D1_bar, D2_bar, A_bar, ...
    resStructFilterAugmented.Z_tilde, resStructFilterAugmented.Finv, resStructFilterAugmented.U, resStructFilterAugmented.a_t_t, resStructFilterAugmented.P_t_t);


%% comparison
disp('Augmented: Smoother 1 (Eq. (4.12)) vs. Smoother 2 (Eq. (4.16))')
max(max(resStructSmoother1Augmented.a_t_T - resStructSmoother2Augmented.a_t_T))
disp('Augmented: Smoother 1 (Eq. (4.12)) vs. Smoother 3 (Eq. (4.3))')
max(max(resStructSmoother1Augmented.a_t_T - resStructSmoother3Augmented.a_t_T))

disp('Augmented: Smoother 1 (Eq. (4.12)) vs. Matlab Build-In Implementation')
max(max(resStructSmoother1Augmented.a_t_T - smoothStatesBuildIn))

disp('Augmented: Smoother 1 (Eq. (4.12)) vs. Nimark Smoother (Eq. (3.2))')
max(max(resStructSmoother1Augmented.a_t_T - resStructNimarkSmootherAugmented.a_t_T))


