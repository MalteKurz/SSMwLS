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

[a_0_0, P_0_0] = initializeSSM(A, C, dimState);
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
resStruct_JKA_Smoother = modifiedDeJongKohnAnsleySmoother(D1, D2, A, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K, resStructFilter.a_t_t, resStructFilter.P_t_t);

resStruct_K_Smoother = modifiedKoopmanSmoother(D1, D2, A, C, R, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K);

resStruct_AM_Smoother = modifiedAndersonMooreSmoother(D1, D2, A, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K, resStructFilter.a_t_t, resStructFilter.P_t_t);

%% comparison
disp('Modified: Smoother 1 (Eq. (4.12)) vs. Smoother 2 (Eq. (4.16))')
max(max(resStruct_JKA_Smoother.a_t_T - resStruct_K_Smoother.a_t_T))
disp('Modified: Smoother 1 (Eq. (4.12)) vs. Smoother 3 (Eq. (4.3))')
max(max(resStruct_JKA_Smoother.a_t_T - resStruct_AM_Smoother.a_t_T))

disp('Modified: Smoother 1 (Eq. (4.12)) vs. Nimark Smoother (Eq. (3.2))')
max(max(resStruct_JKA_Smoother.a_t_T - resStructNimarkSmoother.a_t_T))



%% augmented system and smoothing
A_bar = [A zeros(dimState, dimState); eye(dimState), zeros(dimState, dimState)];
C_bar = [C; zeros(dimState, dimDisturbance)];
D1_bar = [D1, D2];
D2_bar = zeros(dimObs, 2*dimState);

% augmented system with MATLAB build-in
xx = C(1:dimState, 1:dimState);
C_tilde = [xx; zeros(dimState, dimState)];
R_tilde = R(1:dimObs, dimState+1:dimState+dimObs);

mdl = ssm(A_bar, C_tilde, D1_bar, R_tilde);
filteredStatesBuildIn = filter(mdl, Z);
smoothStatesBuildIn = smooth(mdl,Z);

% augmented system smoothing with corrected smoothers
[negLogLike, resStructFilterAugmented] = modifiedFilter(Z, D1_bar, D2_bar, A_bar, C_bar, R);

% Implementation of the Nimark (2015) smoother
resStructNimarkSmootherAugmented = nimarkSmoother(D1_bar, D1_bar, A_bar, ...
    resStructFilterAugmented.Finv, resStructFilterAugmented.U, resStructFilterAugmented.K, resStructFilterAugmented.a_t_t, resStructFilterAugmented.P_t_t, resStructFilterAugmented.P_tp1_t);

resStruct_JKA_SmootherAugmented = modifiedDeJongKohnAnsleySmoother(D1_bar, D2_bar, A_bar, ...
    resStructFilterAugmented.Z_tilde, resStructFilterAugmented.Finv, resStructFilterAugmented.K, resStructFilterAugmented.a_t_t, resStructFilterAugmented.P_t_t);

resStruct_K_SmootherAugmented = modifiedKoopmanSmoother(D1_bar, D2_bar, A_bar, C_bar, R, ...
    resStructFilterAugmented.Z_tilde, resStructFilterAugmented.Finv, resStructFilterAugmented.K);

resStruct_AM_SmootherAugmented = modifiedAndersonMooreSmoother(D1_bar, D2_bar, A_bar, ...
    resStructFilterAugmented.Z_tilde, resStructFilterAugmented.Finv, resStructFilterAugmented.K, resStructFilterAugmented.a_t_t, resStructFilterAugmented.P_t_t);


%% comparison
disp('Augmented: Smoother 1 (Eq. (4.12)) vs. Smoother 2 (Eq. (4.16))')
max(max(resStruct_JKA_SmootherAugmented.a_t_T - resStruct_K_SmootherAugmented.a_t_T))
disp('Augmented: Smoother 1 (Eq. (4.12)) vs. Smoother 3 (Eq. (4.3))')
max(max(resStruct_JKA_SmootherAugmented.a_t_T - resStruct_AM_SmootherAugmented.a_t_T))

disp('Augmented: Smoother 1 (Eq. (4.12)) vs. Matlab Build-In Implementation')
max(max(resStruct_JKA_SmootherAugmented.a_t_T - smoothStatesBuildIn))

disp('Augmented: Smoother 1 (Eq. (4.12)) vs. Nimark Smoother (Eq. (3.2))')
max(max(resStruct_JKA_SmootherAugmented.a_t_T - resStructNimarkSmootherAugmented.a_t_T))


