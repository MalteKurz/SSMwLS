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
    
    properties
        nObs = 2000;
        resStruct % resStruct from the modified filter
        Z
        dimState
    end
    
    
    methods(TestMethodSetup, ParameterCombination='sequential')
        function MethodSetup(testCase, A, C, R, D1, D2)
            
            %% initialize
            [dimObs, testCase.dimState, dimDisturbance] = checkDimsModifiedSSM(D1, D2, A, C, R);
            testCase.Z     = nan(testCase.nObs, dimObs);
            X              = nan(testCase.nObs, testCase.dimState);
            
            [a_0_0, P_0_0] = initializeSSM(A, C, testCase.dimState);
            X(1,:) = mvnrnd(a_0_0, P_0_0);
            
            %% simulate
            u = randn(testCase.nObs, dimDisturbance);
            
            for iObs = 1:testCase.nObs
                X(iObs+1, :) = A * X(iObs,:)' + C * u(iObs,:)';
                testCase.Z(iObs, :)   = D1 * X(iObs+1,:)' + D2 * X(iObs,:)' + R * u(iObs,:)';
                
            end
            
            %% filter
            [~, testCase.resStruct.modifiedFilter] = modifiedFilter(testCase.Z, D1, D2, A, C, R);
            
            %% modified smoother
            testCase.resStruct.AM_Smoother = modifiedAndersonMooreSmoother(D1, D2, A, ...
                testCase.resStruct.modifiedFilter.Z_tilde, testCase.resStruct.modifiedFilter.Finv, testCase.resStruct.modifiedFilter.K, testCase.resStruct.modifiedFilter.a_t_t, testCase.resStruct.modifiedFilter.P_t_t);

            testCase.resStruct.JKA_Smoother = modifiedDeJongKohnAnsleySmoother(D1, D2, A, ...
                testCase.resStruct.modifiedFilter.Z_tilde, testCase.resStruct.modifiedFilter.Finv, testCase.resStruct.modifiedFilter.K, testCase.resStruct.modifiedFilter.a_t_t, testCase.resStruct.modifiedFilter.P_t_t);
            
            testCase.resStruct.K_Smoother = modifiedKoopmanSmoother(D1, D2, A, C, R, ...
                testCase.resStruct.modifiedFilter.Z_tilde, testCase.resStruct.modifiedFilter.Finv, testCase.resStruct.modifiedFilter.K);
            
            %% augmented system
            % get matrices of the augmented system
            augmentedSSM = getAugmentedSystem(D1, D2, A, C, R);
            
            % filter with augmented system
            [~, testCase.resStruct.augmentedFilter] = modifiedFilter(testCase.Z, augmentedSSM.D1, augmentedSSM.D2, augmentedSSM.A, augmentedSSM.C, augmentedSSM.R);
            
            
            %% apply modified smmother to augmented system
            testCase.resStruct.AM_SmootherAugmented = modifiedAndersonMooreSmoother(augmentedSSM.D1, augmentedSSM.D2, augmentedSSM.A, ...
                testCase.resStruct.augmentedFilter.Z_tilde, testCase.resStruct.augmentedFilter.Finv, testCase.resStruct.augmentedFilter.K, testCase.resStruct.augmentedFilter.a_t_t, testCase.resStruct.augmentedFilter.P_t_t);

            testCase.resStruct.JKA_SmootherAugmented = modifiedDeJongKohnAnsleySmoother(augmentedSSM.D1, augmentedSSM.D2, augmentedSSM.A, ...
                testCase.resStruct.augmentedFilter.Z_tilde, testCase.resStruct.augmentedFilter.Finv, testCase.resStruct.augmentedFilter.K, testCase.resStruct.augmentedFilter.a_t_t, testCase.resStruct.augmentedFilter.P_t_t);
            
            testCase.resStruct.K_SmootherAugmented = modifiedKoopmanSmoother(augmentedSSM.D1, augmentedSSM.D2, augmentedSSM.A, augmentedSSM.C, augmentedSSM.R, ...
                testCase.resStruct.augmentedFilter.Z_tilde, testCase.resStruct.augmentedFilter.Finv, testCase.resStruct.augmentedFilter.K);
            
            %% apply Matlab's build-in function to augmented system
            xx = C(1:testCase.dimState, 1:testCase.dimState);
            C_tilde = [xx; zeros(testCase.dimState, testCase.dimState)];
            R_tilde = R(1:dimObs, testCase.dimState+1 : testCase.dimState+dimObs);
            
            mdl = ssm(augmentedSSM.A, C_tilde, augmentedSSM.D1, R_tilde);
            testCase.resStruct.filteredStatesBuildIn = filter(mdl, testCase.Z);
            testCase.resStruct.smoothStatesBuildIn = smooth(mdl, testCase.Z);
            
            
        end
    end
    
    methods (Test)
        function AM_vs_JKA(testCase)
            
            testCase.verifyEqual(testCase.resStruct.AM_Smoother.a_t_T, testCase.resStruct.JKA_Smoother.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function AM_vs_K(testCase)
            
            testCase.verifyEqual(testCase.resStruct.AM_Smoother.a_t_T, testCase.resStruct.K_Smoother.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function JKA_vs_K(testCase)
            
            testCase.verifyEqual(testCase.resStruct.JKA_Smoother.a_t_T, testCase.resStruct.K_Smoother.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function modifiedFilter_vs_augmentedFilter(testCase)
            
            testCase.verifyEqual(testCase.resStruct.modifiedFilter.a_t_t, testCase.resStruct.augmentedFilter.a_t_t(:,1:testCase.dimState),...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function AM__modified_vs_augmented(testCase)
            
            testCase.verifyEqual(testCase.resStruct.AM_Smoother.a_t_T, testCase.resStruct.AM_SmootherAugmented.a_t_T(:,1:testCase.dimState),...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function JKA__modified_vs_augmented(testCase)
            
            testCase.verifyEqual(testCase.resStruct.JKA_Smoother.a_t_T, testCase.resStruct.JKA_SmootherAugmented.a_t_T(:,1:testCase.dimState),...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function K__modified_vs_augmented(testCase)
            
            testCase.verifyEqual(testCase.resStruct.K_Smoother.a_t_T, testCase.resStruct.K_SmootherAugmented.a_t_T(:,1:testCase.dimState),...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function augmented__AM_vs_JKA(testCase)
            
            testCase.verifyEqual(testCase.resStruct.AM_SmootherAugmented.a_t_T, testCase.resStruct.JKA_SmootherAugmented.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function augmented__AM_vs_K(testCase)
            
            testCase.verifyEqual(testCase.resStruct.AM_SmootherAugmented.a_t_T, testCase.resStruct.K_SmootherAugmented.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function augmented__JKA_vs_K(testCase)
            
            testCase.verifyEqual(testCase.resStruct.JKA_SmootherAugmented.a_t_T, testCase.resStruct.K_SmootherAugmented.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function augmentedFilter_vs_MatlabBuildIn(testCase)
            
            testCase.verifyEqual(testCase.resStruct.augmentedFilter.a_t_t, testCase.resStruct.filteredStatesBuildIn,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function augmented__AM_vs_MatlabBuildIn(testCase)
            
            testCase.verifyEqual(testCase.resStruct.smoothStatesBuildIn, testCase.resStruct.AM_SmootherAugmented.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function augmented__JKA_vs_MatlabBuildIn(testCase)
            
            testCase.verifyEqual(testCase.resStruct.smoothStatesBuildIn, testCase.resStruct.JKA_SmootherAugmented.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
        end
        
        function augmented__K_vs_MatlabBuildIn(testCase)
            
            testCase.verifyEqual(testCase.resStruct.smoothStatesBuildIn, testCase.resStruct.K_SmootherAugmented.a_t_T,...
                'AbsTol', 10^-12, 'RelTol', 10^-12);
            
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


