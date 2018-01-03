%% Example: ARMA(1,1) with measurement error as State Space Models with Lagged State (SSMwLS) in the measurement equation
%
% Author: Malte S. Kurz
%
%% Note
%
%      Example taken from Kurz (2018): Section 5 Application: ARMA dynamics
%      with measurement error
%
%% References
%
%      Kurz, M. S. 2018. "A note on low-dimensional Kalman smoother for
%         systems with lagged states in the measurement equation"
%
%      Nimark, K. P. 2015. "A low dimensional Kalman filter for systems
%         with lagged states in the measurement equation". Economics
%         Letters 127: 10-13.
%
%

%% Parametrization
nObs = 5000;                       % number of observations
phi = 0.9;                         % AR(1)-parameter
theta = 0.5;                       % MA(1)-parameter
sigma_eps = 1;                     % standard-deviation of the signal-disturbance
q = 1.5;                           % signal-to-noise ratio 
sigma_delta = sigma_eps / sqrt(q); % standard-deviation of the measurement error

%% Matrices for the SSMwLS
A = phi;
C = [sigma_eps, 0];
R = [0, sigma_delta];
D1 = 1;
D2 = theta;

%% Initialize
[dimObs, dimState, dimDisturbance] = checkDimsModifiedSSM(D1, D2, A, C, R);
Z              = nan(nObs, dimObs);
X              = nan(nObs, dimState);

[a_0_0, P_0_0] = initializeSSM(A, C, dimState);
X(1,:)         = mvnrnd(a_0_0, P_0_0);

%% Simulate
u = randn(nObs, dimDisturbance);

for iObs = 1:nObs
    X(iObs+1, :) = A * X(iObs,:)' + C * u(iObs,:)';
    Z(iObs, :)   = D1 * X(iObs+1,:)' + D2 * X(iObs,:)' + R * u(iObs,:)';
    
end

%% Filter
[~, resStructFilter] = modifiedFilter(Z, D1, D2, A, C, R);

%% Smoother

% Modified Anderson and Moore (1979) smoother (Eq. (4.3) in Kurz (2018))
resStruct_AM_Smoother = modifiedAndersonMooreSmoother(D1, D2, A, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K, resStructFilter.a_t_t, resStructFilter.P_t_t);

% Modified de Jong (1988, 1989) and Kohn and Ansley (1989) smoother (Eq. (4.11) in Kurz (2018))
resStruct_JKA_Smoother = modifiedDeJongKohnAnsleySmoother(D1, D2, A, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K, resStructFilter.a_t_t, resStructFilter.P_t_t);

% Modified Koopman (1993) smoother (Eq. (4.14)-(4.15) in Kurz (2018))
resStruct_K_Smoother = modifiedKoopmanSmoother(D1, D2, A, C, R, ...
    resStructFilter.Z_tilde, resStructFilter.Finv, resStructFilter.K);

% Nimark's (2015) smoother
resStructNimarkSmoother = nimarkSmoother(D1, D2, A, ...
    resStructFilter.Finv, resStructFilter.U, resStructFilter.K, resStructFilter.a_t_t, resStructFilter.P_t_t, resStructFilter.P_tp1_t);


%% Comparison
fprintf(['Modified Anderson and Moore (1979) smoother vs. Modified de Jong (1988, 1989) and Kohn and Ansley (1989) smoother:\n',...
'Max norm of difference: ', num2str(max(max(resStruct_AM_Smoother.a_t_T - resStruct_JKA_Smoother.a_t_T))), '\n\n']);

fprintf(['Modified Anderson and Moore (1979) smoother vs. Modified Koopman (1993) smoother:\n',...
'Max norm of difference: ', num2str(max(max(resStruct_AM_Smoother.a_t_T - resStruct_K_Smoother.a_t_T))), '\n\n']);

fprintf(['Modified de Jong (1988, 1989) and Kohn and Ansley (1989) smoother vs. Nimark''s (2015) smoother:\n',...
'Max norm of difference: ', num2str(max(max(resStruct_JKA_Smoother.a_t_T - resStructNimarkSmoother.a_t_T))), '\n\n']);

