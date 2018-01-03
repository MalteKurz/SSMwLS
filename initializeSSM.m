function [a_0_0, P_0_0] = initializeSSM(A, C, dimState)

a_0_0 = zeros(dimState, 1);

CC = C * C';
xx = eye(dimState*dimState) - kron(A,A);
xx = eye(size(xx)) / xx;
P_0_0 = reshape(xx*CC(:), dimState, dimState);

end
