clc
clear
close all

tic

% Add LIBSVM and NPY package
addpath(genpath('./libsvm-3.17'));
addpath(genpath('./npy-matlab-master')) 

attack_names = {'flipped'}
%attack_names = {'Deepfool', 'CW100', 'CW0', 'FGSM10'}
numbers = {5, 10, 30, 50, 100, 192}
for  numm=1:6
    it_num = numbers{numm}
    file_name_save = sprintf('/home/ehsan/fliipimg_mat_files/2_percent/res_match_index_%d.txt', it_num);
    fid =fopen(file_name_save, 'a+' );
for atk_name=1:1
    test_case = attack_names{atk_name}
    mean_array_tot = 0;
    
    mean_array = 0;
    for model_count=0:49
        model_name = sprintf('/home/ehsan/fliipimg_mat_files/2_percent/%d/results_%d.mat',it_num, model_count);
        model_2CC =load(model_name,'model_2C');
        model_2CC = model_2CC.model_2C;
        
        data_attack_matrix = sprintf('/home/ehsan/Desktop/Mohammad_Label_Flip/RDFS_Flipping/2_Percent/%d/inter_feature_50/StammNet_subset_%s_%d.npy',it_num, test_case,model_count);
        data_attack_matrix = readNPY(data_attack_matrix);
        
        num_test = length(data_attack_matrix);
        nn = num_test;
        num_test = nn;
        te_idx = 1:nn;

        %--------------------------------------------------------------------------
        %                                     Testing
        %--------------------------------------------------------------------------
        %testLabel = [ones(1,500)]'
        testLabel = [ones(1,nn)]';
        X_test = [data_attack_matrix(te_idx, :)];

        % Test
        [predict_label, accuracy, decision_function] = svmpredict(testLabel, X_test, model_2CC, ' -b 0');

        mean_array = mean_array + accuracy(1);
    end

    total_mean = mean_array / 50;
    sprintf('Final Average = %f, %s', total_mean, test_case);
    fprintf(fid, 'Final Average = %f, %s \n', total_mean, test_case);
end
fclose(fid);
end