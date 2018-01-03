classdef UnitTestsSSMwLS < matlab.unittest.TestCase
    
    properties (MethodSetupParameter)
        A = {diag([0.8, 0.2, 0.1]),...
            0.9};
        C = {[diag([1, 0.9, 1.4]), zeros(3,2)],...
            [1, 0]};
        R = {[zeros(2,3), diag([0.8, 1.1])],...
            [0, 1]};
        D1 = {[1, 0.2, 0.1; 0.7, 0.9, 0.2],...
            1};
        D2 = {[0.5, 0.1, 0.05; 0.9, 0.05, 0.2],...
            0.5};
    end
    
    %properties (TestParameter)
    %    smoother = {'modifiedAndersonMooreSmoother', 'modifiedDeJongKohnAnsleySmoother', 'modifiedKoopmanSmoother'}
    %end
    
    properties
        nObs = 2000;
        resStructFilter % resStruct from the modified filter
        parameters
        Z
    end
    
    
    methods(TestMethodSetup, ParameterCombination='sequential')
        function MethodSetup(testCase, A, C, R, D1, D2)
            %% initialize
            [dimObs, dimState, dimDisturbance] = checkDimsModifiedSSM(D1, D2, A, C, R);
            testCase.Z     = nan(testCase.nObs, dimObs);
            X              = nan(testCase.nObs, dimState);
            
            [a_0_0, P_0_0] = initializeSSM(A, C, dimState);
            X(1,:) = mvnrnd(a_0_0, P_0_0);
            
            %% simulate
            u = randn(testCase.nObs, dimDisturbance);
            
            for iObs = 1:testCase.nObs
                X(iObs+1, :) = A * X(iObs,:)' + C * u(iObs,:)';
                testCase.Z(iObs, :)   = D1 * X(iObs+1,:)' + D2 * X(iObs,:)' + R * u(iObs,:)';
                
            end
            
            % filter
            [~, testCase.resStructFilter] = modifiedFilter(testCase.Z, D1, D2, A, C, R);
            
            % add setup parameters to testCase
            testCase.parameters = struct('A', A, 'C', C, 'R', R, 'D1', D1, 'D2', D2);
            
        end
    end
    
    methods (Test)
        function testAllModifiedSmootherEqual(testCase)
            resStruct_AM_Smoother = modifiedAndersonMooreSmoother(testCase.parameters.D1, testCase.parameters.D2, testCase.parameters.A, ...
                testCase.resStructFilter.Z_tilde, testCase.resStructFilter.Finv, testCase.resStructFilter.K, testCase.resStructFilter.a_t_t, testCase.resStructFilter.P_t_t);

            resStruct_JKA_Smoother = modifiedDeJongKohnAnsleySmoother(testCase.parameters.D1, testCase.parameters.D2, testCase.parameters.A, ...
                testCase.resStructFilter.Z_tilde, testCase.resStructFilter.Finv, testCase.resStructFilter.K, testCase.resStructFilter.a_t_t, testCase.resStructFilter.P_t_t);
            
            resStruct_K_Smoother = modifiedKoopmanSmoother(testCase.parameters.D1, testCase.parameters.D2, testCase.parameters.A, testCase.parameters.C, testCase.parameters.R, ...
                testCase.resStructFilter.Z_tilde, testCase.resStructFilter.Finv, testCase.resStructFilter.K);
            
            
            testCase.verifyEqual(resStruct_AM_Smoother.a_t_T, resStruct_JKA_Smoother.a_t_T,...
                'AbsTol', 10^-8, 'RelTol', 10^-8);
            testCase.verifyEqual(resStruct_AM_Smoother.a_t_T, resStruct_K_Smoother.a_t_T,...
                'AbsTol', 10^-8, 'RelTol', 10^-8);
            
        end
        
        function testModifiedFilter_vs_AugmentedFilter(testCase)
            
            % get matrices of the augmented system
            augmentedSSM = getAugmentedSystem(testCase.parameters.D1, testCase.parameters.D2, testCase.parameters.A, testCase.parameters.C, testCase.parameters.R);
            
            % filter with augmented system
            [~, resStructFilterAugmented] = modifiedFilter(testCase.Z, augmentedSSM.D1, augmentedSSM.D2, augmentedSSM.A, augmentedSSM.C, augmentedSSM.R);
            
            dimState= size(testCase.parameters.D1,2);
            
            testCase.verifyEqual(testCase.resStructFilter.a_t_t, resStructFilterAugmented.a_t_t(:,1:dimState),...
                'AbsTol', 10^-8, 'RelTol', 10^-8);
            
        end
        
    end
    
end


function res = getAugmentedSystem(D1, D2, A, C, R)

[dimObs, dimState, dimDisturbance] = checkDimsModifiedSSM(D1, D2, A, C, R);

% augmented system matrices
res.A = [A zeros(dimState, dimState); eye(dimState), zeros(dimState, dimState)];
res.C = [C; zeros(dimState, dimDisturbance)];
res.D1 = [D1, D2];
res.D2 = zeros(dimObs, 2*dimState);

res.R = R;

end


