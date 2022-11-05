      
function [Pre_Labels,Outputs]=LabelSpecificFeature(cv_test_data,cv_test_target,cv_train_data,cv_train_target,model_LLSF,model_MLL)
       if model_MLL.type == 0 % Whether to extract label-specific features yes=1 no=0;是否进行类属属性提取
%             Outputs       = tanh(cv_test_data*model_NetworkML.W1)*model_NetworkML.W2;
%             [model_LLSF]  = LLSF( cv_train_data, cv_train_target',optmParameter);
            Outputs       = cv_test_data*model_LLSF;
           % Outputs       = cv_test_data*model_LLSF.W + model_LLSF.b;
            fscore                 = cv_train_data*model_LLSF;
            %fscore                 = cv_train_data*model_LLSF.W + model_LLSF.b;
            [ tau,  currentResult] = TuneThreshold( fscore', cv_train_target, 1, 1);
            Pre_Labels             = Predict(Outputs',tau);
       elseif model_MLL.type == 1
           %% str SVM  as a Feature Selection Method
            [num_labels,num_train]=size(cv_train_target);
            svm.type='Linear';
            svm.para=[];
            switch svm.type
                case 'RBF'
                    gamma=num2str(svm.para);
                    str=['-t 2 -g ',gamma,' -b 1'];
                case 'Poly'
                    gamma=num2str(svm.para(1));
                    coef=num2str(svm.para(2));
                    degree=num2str(svm.para(3));
                    str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
                case 'Linear'
                    str='-t 0 -b 1 -q';
                otherwise
                     error('SVM types not supported, please type "help LIFT" for more information');

            end
             num_test=size(cv_test_target,2);
             Pre_Labels=[];
             Outputs=[];

             for ii=1:num_labels
                idx_feature = (model_LLSF(:,ii)~=0);
                training_label_vector=cv_train_target(ii,:)';

                Models{ii,1}=svmtrain(training_label_vector,cv_train_data(:,idx_feature),str);     

                testing_label_vector=cv_test_target(ii,:)';

                [predicted_label,accuracy,prob_estimates]=svmpredict(testing_label_vector,cv_test_data(:,idx_feature),Models{ii,1},'-b 1');

                if(isempty(predicted_label))
                    predicted_label=cv_train_target(ii,1)*ones(num_test,1);
                    if(cv_train_target(ii,1)==1)
                        Prob_pos=ones(num_test,1);
                    else
                        Prob_pos=zeros(num_test,1);
                    end
                    Outputs=[Outputs;Prob_pos'];
                    Pre_Labels=[Pre_Labels;predicted_label'];
                else
                    pos_index=find(Models{ii,1}.Label==1);
                    Prob_pos=prob_estimates(:,pos_index);
                    Outputs=[Outputs;Prob_pos'];
                    Pre_Labels=[Pre_Labels;predicted_label'];
                end
             end
%              cv_test_target=cv_test_target';
                Outputs=Outputs';
       end
   end