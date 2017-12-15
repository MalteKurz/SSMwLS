function [LL]=logl(A,C,D1,D2,R,Z)
if D2==0;
    D2=D1*0;
end
%Computes the recursive log likelihood LL for systems with lagged
%observables taking the unconditional state variance as initial state
%covariance

%Inputs are matrices of state space system
% 	X[t] = AX[t-1] + Cu[t]
% 
%   Z[t] = D1X[t] + D2X[t-1] + Ru[t]

LL=0;
T=length(Z);
dimX=length(A);
dimZ=length(D1(:,1));
CC=C*C';
P0=reshape(inv(eye(dimX*dimX)-kron(A,A))*CC(:),dimX,dimX); %compute E(XtXt')

Xhat=zeros(dimX,T+1);
for tt=1:T
    Ztilde=Z(:,tt)-D1*A*Xhat(:,tt)-D2*Xhat(:,tt);
    Omega=(D1*A+D2)*P0*(D1*A+D2)'+(D1*C+R)*(D1*C+R)';
    Omegainv=eye(dimZ)/Omega;
    K=(A*P0*(D1*A+D2)'+C*C'*D1'+C*R')*Omegainv;
    Xhat(:,tt+1)=A*Xhat(:,tt)+K*Ztilde;
    P1=A*P0*A'+C*C';
    P0=P1-K*Omega*K';
    LL = LL - 0.5*(log(det(Omega) + Ztilde'*Omegainv*Ztilde));
end

    LL=LL-0.5*dimZ*T*log(2*pi);


    
    
    