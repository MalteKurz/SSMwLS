function [X]=smooth(A,C,D1,D2,R,Z)
if D2==0;
    D2=D1*0;
end
    
%  Kalman simulation smoother
%  Adapted from Durbin and |Koopman (2002) by K Nimark

%     Xt = A*Xt-1 +C*ut
%
%     Zt=  D1*Xt + D2*Xt-1 + R*ut


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define a ancilliary variables, predefine matrices etc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=length(Z);
dimX=length(A);        %dimension of state
dimZ=length(D1(:,1));   %dimension of observables;

CC=C*C';
P0=reshape(inv(eye(dimX*dimX)-kron(A,A))*CC(:),dimX,dimX);
Xhat=zeros(dimX,T+1);
PP0=zeros(dimX,dimX,T);
PP1=zeros(dimX,dimX,T);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% forward recursion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for tt=1:T
    Ztilde=Z(:,tt)-D1*A*Xhat(:,tt)-D2*Xhat(:,tt);
    Omega=(D1*A+D2)*P0*(D1*A+D2)'+(D1*C+R)*(D1*C+R)';
    Omegainv=eye(dimZ)/Omega;
    K=(A*P0*(D1*A+D2)'+C*C'*D1'+C*R')*Omegainv;
    Xhat(:,tt+1)=A*Xhat(:,tt)+K*Ztilde;
    P1=A*P0*A'+C*C';
    P0=P1-K*Omega*K';
    P1=A*P0*A'+C*C';
    PP0(:,:,tt)=P0;
    PP1(:,:,tt)=P1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%construct what is needed for last step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xtilde=A*Xhat(:,1:end);
Xhat=Xhat(:,1:end);

Xsm=Xhat(:,1:end)*0;
Xsm(:,T+1)=Xtilde(:,end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%backward recursion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for tt=T:-1:1
    J=PP0(:,:,tt)*A'*inv(PP1(:,:,tt)+eye(dimX)*1e-8);
    Xsm(:,tt)=Xhat(:,tt+1)+J*(Xsm(:,tt+1)-Xtilde(:,tt+1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%spit out
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=Xsm;
