function [dimObs, dimState, dimDisturbance] = checkDimsModifiedSSM(D1, D2, A, C, R)

% check dimensions of observables and states
assert(isequal(size(D1), size(D2)))
assert(size(A,1) == size(A,2))

% extract dimensions
[dimObs, dimState] = size(D1);

if not(isempty(C)) && not(isempty(R))
    % check and extract disturbance dimensions
    dimDisturbance = size(C,2);
    assert(size(C,2) == size(R,2))
elseif not(isempty(C)) && isempty(R)
    % extract disturbance dimensions
    dimDisturbance = size(C,2);
elseif not(isempty(R)) && isempty(C)
    % extract disturbance dimensions
    dimDisturbance = size(R,2);
else
    % nothing to do
end

end