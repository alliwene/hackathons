def fit_predict(model, X, y, K, test_df, plot=True, sampling=False):
    # set initial scores
    scores = 0
    auc_scores = 0
    # set empty list to store predictions on test set
    test_oofs = []
    # get model name
    model_name = type(model).__name__
    train_accuracies, test_accuracies = [0.5], [0.5]

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
            # under = RandomUnderSampler(random_state=1)
            # steps = [('o', over), ('u', under)]
            # pipeline = Pipeline(steps=steps)

            # resample train dataset
            X_train, y_train = over.fit_resample(X_train, y_train)

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
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        # predict on train test
        train_prob = model.predict_proba(X_train)[:, 1]
        # get F1-score and roc_auc_score
        score = f1_score(y_test, pred, average='macro')
        roc = roc_auc_score(y_test, prob)
        train_roc = roc_auc_score(y_train, train_prob)
        # append roc-auc for train and test
        train_accuracies.append(train_roc)
        test_accuracies.append(roc)
        # take mean of scores
        scores += score/K
        auc_scores += roc/K
        test_oofs.append(pred)

        if i % 4 == 0:
            print('Fold {} F1-score: {}'.format(i+1, score))
            # print('Fold {} ROC-AUC score: {}'.format(i+1, roc))
            print('='*45)

    print()
    print('Avg F1 score: {:.4f} '.format(scores))
    print('Avg ROC-AUC score: {:.4f} '.format(auc_scores))
        
    if plot:
        # plot train and test roc-auc
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label="train roc-auc")
        plt.plot(test_accuracies, label="test roc-auc")
        plt.legend(loc="lower right", prop={'size': 12})
        plt.xticks(range(0, K, 5))
        plt.xlabel("fold", size=12)
        plt.ylabel("roc-auc", size=12)
        plt.show() 
    # make prediction on test set for submission
    test_df = scaler.transform(test_df)
    predictions = model.predict(test_df)

    Model.append(model)
    F1Score.append(scores) 

    return model, predictions  





def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)