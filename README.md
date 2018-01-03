### SSMwLS
---
State Space Model with Lagged State (SSMwLS) in the measurement equation

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/MalteKurz/SSMwLS/blob/master/LICENCE.txt)

### Motivation
This repo contains accompanying code to the paper "A note on low-dimensional Kalman smoother for systems with lagged states in the measurement equation" (Kurz, 2018).

### SSMwLS Model Equations
Measurement Equation

![img](http://latex.codecogs.com/svg.latex?Z_%7Bt%7D%3DD_1X_%7Bt%7D%2BD_%7B2%7DX_%7Bt-1%7D%2BRu_%7Bt%7D)

State Equation

![img](http://latex.codecogs.com/svg.latex?X_%7Bt%7D%26%3DAX_%7Bt-1%7D%2BCu_%7Bt%7D)


### References
Kurz, M. S. 2018. "A note on low-dimensional Kalman smoother for systems with lagged states in the measurement equation"

[Nimark, K. P. 2015. "A low dimensional Kalman filter for systems with lagged states in the measurement equation". Economics Letters 127: 10-13.](https://doi.org/10.1016/j.econlet.2014.12.016)

### Author
Malte S. Kurz


---


### Example
ARMA(1,1)-process with measurement error as State Space Model with Lagged State (SSMwLS) in the measurement equation


#### Note
Example taken from Section 5 of Kurz (2018): "Application: ARMA dynamics with measurement error"



#### Parametrization
```Matlab
nObs = 5000;                       % number of observations
phi = 0.9;                         % AR(1)-parameter
theta = 0.5;                       % MA(1)-parameter
sigma_eps = 1;                     % standard-deviation of the signal-disturbance
q = 1.5;                           % signal-to-noise ratio 
sigma_delta = sigma_eps / sqrt(q); % standard-deviation of the measurement error
```

#### Matrices for the SSMwLS
```Matlab
A = phi;
C = [sigma_eps, 0];
R = [0, sigma_delta];
D1 = 1;
D2 = theta;
```

#### Initialize
```Matlab
[dimObs, dimState, dimDisturbance] = checkDimsModifiedSSM(D1, D2, A, C, R);
Z              = nan(nObs, dimObs);
X              = nan(nObs, dimState);

[a_0_0, P_0_0] = initializeSSM(A, C, dimState);
X(1,:)         = mvnrnd(a_0_0, P_0_0);
```

#### Simulate
```Matlab
u = randn(nObs, dimDisturbance);

for iObs = 1:nObs
    X(iObs+1, :) = A * X(iObs,:)' + C * u(iObs,:)';
    Z(iObs, :)   = D1 * X(iObs+1,:)' + D2 * X(iObs,:)' + R * u(iObs,:)';
    
end
```

#### Filter
```Matlab
[~, resStructFilter] = modifiedFilter(Z, D1, D2, A, C, R);
```

#### Smoother
```Matlab
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
```

#### Comparison
```Matlab
fprintf(['Modified Anderson and Moore (1979) smoother vs. Modified de Jong (1988, 1989) and Kohn and Ansley (1989) smoother:\n',...
'Max norm of difference: ', num2str(max(max(resStruct_AM_Smoother.a_t_T - resStruct_JKA_Smoother.a_t_T))), '\n\n']);

fprintf(['Modified Anderson and Moore (1979) smoother vs. Modified Koopman (1993) smoother:\n',...
'Max norm of difference: ', num2str(max(max(resStruct_AM_Smoother.a_t_T - resStruct_K_Smoother.a_t_T))), '\n\n']);

fprintf(['Modified de Jong (1988, 1989) and Kohn and Ansley (1989) smoother vs. Nimark''s (2015) smoother:\n',...
'Max norm of difference: ', num2str(max(max(resStruct_JKA_Smoother.a_t_T - resStructNimarkSmoother.a_t_T))), '\n\n']);
```
```
Modified Anderson and Moore (1979) smoother vs. Modified de Jong (1988, 1989) and Kohn and Ansley (1989) smoother:
Max norm of difference: 8.8818e-16

Modified Anderson and Moore (1979) smoother vs. Modified Koopman (1993) smoother:
Max norm of difference: 3.5527e-15

Modified de Jong (1988, 1989) and Kohn and Ansley (1989) smoother vs. Nimark's (2015) smoother:
Max norm of difference: 0.35082
```
