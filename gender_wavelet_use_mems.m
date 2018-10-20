lambda = 128;
p = 40;
load('gender_labels.mat');
class_labels = gender_labels;
subject_sample_size = 1;
number_of_subjects_each_fold = 101;
number_of_subjects = 808;

subject_sample_size = 1;

for level = 1:6
    all_folds_mv_predicted_labels= [];
    
    fsg_accs =zeros(8,1);
    all_predicted_labels = [];
    all_task_acc = [];
    for i = 1:8
        
        te_inds = (i-1) * number_of_subjects_each_fold + 1: i * number_of_subjects_each_fold ;
        tr_inds = setdiff(1:number_of_subjects, te_inds);
        
        test_second_probs = [];
        all_train_second_probs = [];
        task_acc = [];
        fold_mv_predicted_labels = [];
        
        for cl = 1:7
            
            all_train_probs=[];
            all_test_probs = [];
            for ll = 1:level
                load(['gender_fsg_memberships_wavelet/fold' num2str(i) '/level' num2str(ll) '/cl' num2str(cl) '/app_memberships.mat']);
                
                all_train_probs = [all_train_probs train_probs];
                all_test_probs = [all_test_probs test_probs];

            end
            wavelet_range = 1:(2*level);
            
            train_second_probs = [];
            for j = 1:7
                
                val_inds = (j-1)*subject_sample_size * number_of_subjects_each_fold + 1: j*subject_sample_size * number_of_subjects_each_fold ;
                tr_tr_inds = setdiff(1:size(all_train_probs,1),val_inds);
                
                model = train_linear(train_labels(tr_tr_inds), sparse(all_train_probs(tr_tr_inds,wavelet_range)), ['-c  1,  -s 0 ' ] );
                [predicted_label,accuracy_te,prob] = predict_linear(train_labels(val_inds), sparse(all_train_probs(val_inds,wavelet_range)), model, ' -b 1 ' );
                x = train_labels(tr_tr_inds);
                if x(1) == 1
                    train_second_probs = [train_second_probs; prob];
                elseif x(1) == 2
                    train_second_probs = [train_second_probs; prob(:,2) prob(:,1)];
                end
            end
            
            all_train_second_probs = [all_train_second_probs train_second_probs];
            
            model = train_linear(train_labels, sparse(all_train_probs(:,wavelet_range)), ['-c  1,  -s 0 ' ] );
            [predicted_label,accuracy_te,prob] = predict_linear(test_labels, sparse(all_test_probs(:,wavelet_range)), model, ' -b 1 ' );
            task_acc = [task_acc accuracy_te(1)];
            x = train_labels;
            if x(1) == 1
                test_second_probs = [test_second_probs prob];
            elseif x(1) == 2
                test_second_probs = [test_second_probs prob(:,2) prob(:,1)];
            end
            fold_mv_predicted_labels = [fold_mv_predicted_labels predicted_label];
            
        end
        
        all_folds_mv_predicted_labels = [all_folds_mv_predicted_labels;fold_mv_predicted_labels];
        
        all_task_acc = [all_task_acc;task_acc];
        model = train_linear(train_labels, sparse(all_train_second_probs), ['-c  1,  -s ' num2str(s) ] );
        [predicted_label,accuracy_te,prob] = predict_linear(test_labels, sparse(test_second_probs), model, ' -b 1 ' );
        all_predicted_labels = [all_predicted_labels;predicted_label];
        
        fsg_accs(i) =accuracy_te(1);
    end
    
    mkdir(['gender_fsg_results_wavelet/level' num2str(level) '/']);
    save(['gender_fsg_results_wavelet/level' num2str(level) '/results.mat'], 'fsg_accs', 'all_predicted_labels','all_task_acc', 'all_folds_mv_predicted_labels');
    
end



