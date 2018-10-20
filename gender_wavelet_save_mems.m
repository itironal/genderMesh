
%%% add liblinear to this path.
lambda = 128; 
p = 40;

load('gender_labels.mat');
class_labels = gender_labels;
number_of_subjects_each_fold = 101;
number_of_subjects = 808;
subject_sample_size = 1;

types = {'app', 'det'};
max_level = 6; %% change this between 1 to 6

fsg_accs =zeros(8,1);
all_predicted_labels = [];
all_wavelet_acc = [];
for i = 1:8 %%  number of folds
    
    test_second_probs = [];
    test_second_predicted_labels = [];
    all_train_second_probs = [];
    all_train_second_predicted_labels = [];
    wavelet_acc = [];
    for level = 1:max_level
        disp(['fold = ' num2str(i) ', level = ' num2str(level)]);
        
        for t=1:2 %% approximation or detail part
            
            if t==1
                load(['combined_weights/level' num2str(level) '/p' num2str(best_p) '/lambda' num2str(best_lambda) '/combined_weights.mat'])
            elseif t== 2
                load(['combined_weights/level' num2str(level) '/p' num2str(best_p) '/lambda' num2str(best_lambda) '/combined_weights.mat'])
            end
                        
            if t==1
                all_subjects_a = all_subjects_a_app;
            elseif t==2
                all_subjects_a = all_subjects_a_det;
            end
                        
            exp = {};
            for cl = 1:7
                exp = [exp all_subjects_a(cl:7:end,:)];
            end
            
            te_inds = (i-1) * number_of_subjects_each_fold + 1: i * number_of_subjects_each_fold ;
            tr_inds = setdiff(1: number_of_subjects, te_inds);
            
            all_train_probs = [];
            all_train_predicted_labels = [];
            train_labels = [];
            
            for cl = 1:7
                train_probs = [];
                train_predicted_labels = [];
                for j = 1:7
                                     
                    val_inds = tr_inds((j-1)*subject_sample_size * number_of_subjects_each_fold + 1: j*subject_sample_size * number_of_subjects_each_fold) ;
                    tr_tr_inds = setdiff(tr_inds,val_inds);
                  
                    model = train_linear(class_labels(tr_tr_inds), sparse(exp{cl}(tr_tr_inds,:)), ['-c  1,  -s 0 ' ] );
                    [predicted_label,accuracy_te,prob] = predict_linear(class_labels(val_inds), sparse(exp{cl}(val_inds,:)), model, ' -b 1 ' );
                    x = class_labels(tr_tr_inds);
                    if x(1) == 1
                        train_probs = [train_probs;prob];
                    elseif x(1) == 2
                        train_probs = [train_probs;prob(:,2) prob(:,1)];
                    end
                    train_predicted_labels = [train_predicted_labels;predicted_label];
                    
                end
                
                train_labels = class_labels(tr_inds);
                
                model = train_linear(class_labels(tr_inds), sparse(exp{cl}(tr_inds,:)), ['-c  1,  -s 0 ' ] );
                [predicted_label,accuracy_te,prob] = predict_linear(class_labels(te_inds), sparse(exp{cl}(te_inds,:)), model, ' -b 1 ' );
                x = class_labels(tr_inds);
                if x(1) == 1
                    test_probs =  prob;
                elseif x(1) == 2
                    test_probs = [prob(:,2) prob(:,1)];
                end
                test_predicted_labels = predicted_label;
                test_labels = class_labels(te_inds);
                mkdir(['gender_fsg_memberships_wavelet/fold' num2str(i) '/level' num2str(level) '/cl' num2str(cl)]);
                save(['gender_fsg_memberships_wavelet/fold' num2str(i) '/level' num2str(level) '/cl' num2str(cl) '/' types{t} '_memberships.mat']  ,'train_probs', 'test_probs', 'train_labels', 'test_labels',  'train_predicted_labels',  'test_predicted_labels');      
            end           
        end
    end
end

