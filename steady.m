function [K,P]=steady(A,C,D1,D2,R)
if D2==0;
    D2=D1*0;
end
%Computes the Kalman filter for systems with lagged observables
%Inputs are matrices of state space system
% 	X[t] = AX[t-1] + Cu[t]
% 
%   Z[t] = D1X[t] + D2X[t-1] + Ru[t]
%
% Outputs are the Kalman gain K in 
% X[t|t] = X[t|t-1]+ K(Z[t] - D1X[t|t-1] + D2X[t-1|t-1])
%
% and P is the steady state prior error covariance matrix
%
%  P = E(X[t]-X[t|t-1])(X[t]-X[t|t-1])'

tol=1e-9; %Set convergence criteria
maxiter=1000; % Maximum number of iterations
diff=1; 
iter=1;
P0=C*C';
P1=A*P0*A'+C*C';

if D2==0;
    D2=D1*0;
end

while diff>= tol && iter <= maxiter 
   L=(D1*A+D2)*P0*(D1*A+D2)'+(D1*C+R)*(D1*C+R)';
   K=(A*P0*(D1*A+D2)'+C*C'*D1'+C*R')/(L);
   P0=P1-K*L*K';
   P1st=A*P0*A'+C*C';
   diff=max(max(abs(P1-P1st)));
   iter=iter+1;
   P1=P1st;
end
if iter >=maxiter
    display('Max number of iterations achieved without convergence');
end
P=P1;
 