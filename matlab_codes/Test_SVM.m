

clc

clear

close all



tic



% Add LIBSVM and NPY package

addpath(genpath('./libsvm-3.17'));

addpath(genpath('./npy-matlab-master')) 


attack_names = {'flipped'}

%attack_names = {'FGSM10'}

numbers = {5, 10, 30, 50, 100, 192}

for  numm=1:6

    it_num = numbers{numm}

    file_name_save = sprintf('/home/ehsan/fliipimg_mat_files/10_percent/result_%d.txt', it_num);

    fid =fopen(file_name_save, 'a+');

for atk_name=1:1

    test_case = attack_names{atk_name}

    mean_array_tot = 0;

    nn = 0;


    for model_count=0:49

    model_name = sprintf('/home/ehsan/fliipimg_mat_files/10_percent/%d/results_%d.mat',it_num, model_count);

    model_2CC =load(model_name,'model_2C');

    model_2CC = model_2CC.model_2C;

    mean_array = 0;

        for jj=0:49

        data_attack_matrix = sprintf('/home/ehsan/Desktop/Mohammad_Label_Flip/RDFS_Flipping/10_Percent/%d/inter_feature_50/StammNet_subset_%s_%d.npy',it_num, test_case,jj);

        data_attack_matrix = readNPY(data_attack_matrix);

        

        num_test = length(data_attack_matrix);
        nn = num_test;

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

        % % % %---------------------------------------------------------------------

        % % % %                       Save The Results

        % % % %---------------------------------------------------------------------

        % % % 

        %results=sprintf('IFGSM10_results_%d.mat',jj);

        %save([results],'accuracy')

        end

    

    total_mean = mean_array / 50;

    %sprintf('total mean model %d = %f',model_count, total_mean)

    mean_array_tot = mean_array_tot + total_mean;

    end



    final_avg = mean_array_tot / 50;

    sprintf('Final Average = %f, %s', final_avg, test_case);

    fprintf(fid, 'Final Average = %f, %s \n', final_avg, test_case);

end

fclose(fid);

end