# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:06:24 2022

@author: tahamansouri
"""
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import yaml

# user defined libraries
import utils
import TF_Model

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)

def initialize_models():
    '''
    INSTANTIATE MODEL(S)
    COMPILING MODEL(S)
    '''
    models = []
    functions = {}
    functions['rf']=[300]
    functions['rnn']=[configs['T'],configs['D'],configs['hidden_dim'],configs['output_dim'],configs['mid_features']]
    functions['lstm']=[configs['T'],configs['D'],configs['hidden_dim'],configs['output_dim'],configs['mid_features']]
    functions['gru']=[configs['T'],configs['D'],configs['hidden_dim'],configs['output_dim'],configs['mid_features']]
    functions['cnn']=[configs['T'],configs['D'], configs['output_dim'],configs['mid_features']]
    functions['lstm_gmp']=[configs['T'],configs['D'],configs['hidden_dim'],configs['output_dim'],configs['mid_features']]
    functions['cnn_lstm']=[configs['T'],configs['D'], configs['hidden_dim'],configs['output_dim'],configs['mid_features']]
    functions['cnn_gru']=[configs['T'],configs['D'], configs['hidden_dim'], configs['output_dim'],configs['mid_features']]
    for key in functions:
        model = []
        model.append(key)
        model.append(TF_Model.stubFunctions[key](*functions[key]))
        models.append(model)
    for model in models:
        if model[0] !='rf':
            mc = ModelCheckpoint(model[0]+'.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=0)
            model[1].compile(loss='binary_crossentropy',optimizer=Adam(lr=configs['learning_rate']),metrics=['accuracy'])
            model.append(mc)
        else:
            model.append('') 
    return models

def train_models(models,X_train,Y_train):
    for model in models:
        if model[0] =='rf':
            print(model[0], 'training')
            model[1].fit(X_train.reshape(X_train.shape[0],configs['T']), Y_train)
            model.append('')
        else:
            print(model[0], 'training')
            r = model[1].fit(X_train, Y_train, epochs=30, batch_size=64,validation_data = (X_test, Y_test),callbacks=[model[2]], verbose=0)
            model[1] = load_model(model[0]+'.h5')
            model.append(r)
    
    
    # Plot accuracy per iteration
    plt.title('Validation accuracy for length '+str(configs['T']))
    for model in models:
        if model[0] !='rf':
            plt.plot(model[3].history['val_accuracy'], label=model[0]+'_val_acc')
    plt.legend()
    plt.show()
    
    # Plot accuracy per iteration
    plt.title('Training accuracy for length '+str(configs['T']))
    for model in models:
        if model[0] !='rf':
            plt.plot(model[3].history['accuracy'], label=model[0]+'_acc')
    plt.legend()
    plt.show()
    return models

def load_models(models,X_train,Y_train):
    for model in models:
        if model[0] =='rf':
            model[1].fit(X_train.reshape(X_train.shape[0],configs['T']), Y_train)
        else:
            model[1] = load_model(model[0]+'.h5')
    return models

def validation(models,X_test, Y_test):
    print('Validating with',configs['T'],'length.')
    for model in models:
        auc = TF_Model.prediction_test(model[1],model[0],X_test, Y_test)
        model.append(auc)
        
    plt.title('ROC for length '+str(configs['T']))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for model in models:
        if model[0] =='rf':
            fpr, tpr, _ = metrics.roc_curve(Y_test,  model[1].predict(X_test.reshape(X_test.shape[0],X_test.shape[1])))
        else:
            fpr, tpr, _ = metrics.roc_curve(Y_test, model[1](X_test))
        if configs['Train_Models']:
            plt.plot(fpr, tpr,label=model[0]+ " RoC, AUC="+str(model[4]))
        else:
            plt.plot(fpr, tpr,label=model[0]+ " RoC, AUC="+str(model[3]))
            
    plt.plot(fpr,fpr,ls=('dashed'))
    plt.legend()
    plt.show()
    return models

def test_invase(baseline, X_train,Y_train,X_test,Y_test):
    # testing invase model
    print('the selected model is',baseline[0])
    input=X_train.shape[1]
    TF_Model.landa = configs['invase_lambda']
    critic = TF_Model.create_LSTM(configs['T'],configs['D'],configs['hidden_dim'],configs['output_dim'],configs['mid_features'])
    critic.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0001), metrics=['accuracy'])
    actor = TF_Model.create_actor(input, input,configs['T'])
    actor.compile(loss=TF_Model.my_loss, optimizer=optimizers.Adam(0.0001))
    actor,critic = TF_Model.train(X_train,Y_train, configs['batch_size'], actor,critic,baseline[1],configs['epoch_number'],False,X_test,Y_test)
    invase,invase_auc = TF_Model.test_explanation(actor,critic,X_test,Y_test)
    print('# of features',invase[:,configs['T']:2*configs['T']].sum()/invase.shape[0])
    np.savetxt('invase_result.csv', invase, delimiter=',')
    return actor,critic
    
def test_l2x(baseline, X_train,Y_train,X_test,Y_test):
    # testing l2x
    l2x_critic, l2x_actor = TF_Model.L2X(X_train,Y_train, configs['batch_size'],baseline[1],configs['epoch_number'],configs['l2x_feature_no'],configs['T'],X_test, Y_test, configs['hidden_dim'],configs['mid_features'],configs['output_dim'],configs['learning_rate'])
    l2x, l2x_auc = TF_Model.test_gumble(l2x_actor,l2x_critic,X_test,Y_test)
    np.savetxt('l2x_result.csv', l2x, delimiter=',')
    return l2x_critic

def test_gsx(baseline, X_train,Y_train,X_test,Y_test):
    # testing gsx
    TF_Model.landa = configs['gsx_lambda']
    gsx_critic, gsx_actor = TF_Model.GSX(X_train,Y_train, configs['batch_size'],baseline[1],configs['epoch_number'],configs['T'],X_test, Y_test, configs['hidden_dim'],configs['mid_features'],configs['output_dim'],configs['learning_rate'])
    gsx, gsx_auc = TF_Model.test_gumble(gsx_actor,gsx_critic,X_test,Y_test)
    print('# of features',gsx[:,configs['T']:2*configs['T']].sum()/gsx.shape[0])
    np.savetxt('gsx_result.csv', gsx, delimiter=',')
    return gsx_critic
    
def draw_roc(baseline,X_test,Y_test,actor,critic,l2x_critic,gsx_critic):
    ### baseline roc
    baseline_pred_proba = baseline[1](X_test)
    baseline_fpr, baseline_tpr, _ = metrics.roc_curve(Y_test, baseline_pred_proba)
    ### invase roc
    gen_prob=actor(X_test)
    sel_prob = TF_Model.Sample_M(gen_prob)
    invase_pred_probe = critic.predict(X_test*sel_prob[:, :, np.newaxis])
    invase_fpr, invase_tpr, _ = metrics.roc_curve(Y_test, invase_pred_probe)
    ### l2x roc
    l2x_pred_probe=l2x_critic(X_test)
    l2x_fpr,l2x_tpr, _ = metrics.roc_curve(Y_test, l2x_pred_probe)
    ### gsx roc
    gsx_pred_probe=gsx_critic(X_test)
    gsx_fpr,gsx_tpr, _ = metrics.roc_curve(Y_test, gsx_pred_probe)
    #create ROC curve
    plt.title('ROC for length '+str(configs['T']))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(baseline_fpr, baseline_tpr,ls=('dashed'))
    plt.plot(baseline_fpr, baseline_tpr, label="baseline RoC")
    plt.plot(invase_fpr, invase_tpr,label="invase RoC")
    plt.plot(l2x_fpr,l2x_tpr,label="L2X, RoC")
    plt.plot(gsx_fpr,gsx_tpr,label="GSX, RoC")
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    # Loading data and preparing datasets
    dataSets = utils.dataset('first') 
    X_train, Y_train, X_test, Y_test, Y_real_train, Y_real_test = utils.data_prepare(dataSets,'Value',configs['T'], configs['D'],configs['errN'],configs['errThresh'],configs['training_ratio'])
    del dataSets
    print("The training set contains", Y_train.shape[0], "sequences in which there are", int(Y_train[Y_train==1].sum().tolist()), "fault annotations and",Y_train.shape[0]-int(Y_train[Y_train==1].sum().tolist()), "normal ones.")
    print("The test set contains", Y_test.shape[0], "sequences in which there are", int(Y_test[Y_test==1].sum().tolist()), "fault annotations and",Y_test.shape[0]-int(Y_test[Y_test==1].sum().tolist()), "normal ones.")
    
    # Initializing models
    models = initialize_models()
    
    # Training/ loading all predictors
    if configs['Train_Models']:
        models = train_models(models,X_train,Y_train)
    else:
        models = load_models(models,X_train,Y_train)
        
    # Validating all predictors
    models = validation(models,X_test, Y_test)
    baseline=models[configs['best_model']]
    
    # explanation
    actor,critic= test_invase(baseline, X_train,Y_train,X_test,Y_test)
    l2x_critic= test_l2x(baseline, X_train,Y_train,X_test,Y_test)
    gsx_critic= test_gsx(baseline, X_train,Y_train,X_test,Y_test)
    
    # drawing rocs
    draw_roc(baseline,X_test,Y_test,actor,critic,l2x_critic,gsx_critic)
    
    
    
    
    

    
    
	