def fit_predict(model, X, y, K, test_df, sampling=False):
    # set empty list to store predictions on test set
    oofs = np.zeros((len(X)))
    preds = np.zeros((len(test_df)))
    # get model name
    model_name = type(model).__name__
    # train_f1, test_f1 = [0.5], [0.5] 

    # initiate StratifiedKFold
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        # scale the training dataset
        X_train = scaler.transform(X_train)
        # scale the test dataset
        X_test = scaler.transform(X_test)

        if sampling:
            # oversampling and undersampling 
            # define pipeline
            over = SMOTE(sampling_strategy={1:280, 0:1328}, random_state=50)
            under = RandomUnderSampler(sampling_strategy={1:280, 0:1328}, random_state=1)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)

            # resample train dataset
            X_train, y_train = pipeline.fit_resample(X_train, y_train)

        # training
        if model_name in ['VotingClassifier', 'RandomForestClassifier', 
                          'SVC', 'GradientBoostingClassifier']:
            model.fit(X_train, y_train) 
        else:
            model.fit(X_train, y_train,
                      early_stopping_rounds=300,
                      eval_set=[(X_test, y_test)],
                      verbose=False)
            
        # predicting on test set
        prob = model.predict_proba(X_test)[:, 1]
        prob_df = pd.DataFrame(prob)
        prob_df[0] = np.where(prob_df[0]>0.5, 1, 0)  
        train_score = f1_score(y_test, prob_df[0]) 
        # predict on train test
        test_df = scaler.transform(test_df) 
        test_prob = model.predict_proba(test_df)[:, 1] 

        oofs[test_index] = prob 
        preds += test_prob/K 
        
        # append roc-auc for train and test
        # train_f1.append(train_score) 
        
        if i % 4 == 0:
            print(f'Fold {i+1} F1-score: {train_score}') 
            print('='*45)

    print()
    print('Avg F1 score: {:.4f} '.format(preds))
    a = pd.DataFrame(oofs)
    a[0] = np.where(a[0]>0.5, 1, 0)
    oof_score = f1_score((y), (a[0]))
    print(f'\nOOF F1 score is : {oof_score}')
        
    # if plot:
    #     # plot train and test roc-auc
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(train_accuracies, label="train roc-auc")
    #     plt.plot(test_accuracies, label="test roc-auc")
    #     plt.legend(loc="lower right", prop={'size': 12})
    #     plt.xticks(range(0, K, 5))
    #     plt.xlabel("fold", size=12)
    #     plt.ylabel("roc-auc", size=12)
    #     plt.show() 

    Model.append(model)
    F1Score.append(oof_score) 

    return model, preds  