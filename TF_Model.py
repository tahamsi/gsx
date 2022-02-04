import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, GlobalMaxPool1D,Conv1D,MaxPooling1D,GlobalMaxPooling1D,Multiply
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.python.keras.layers import Layer 
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import load_model

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import time




"""
0 General: seven different deep learning models have been created along with 
a method for prediction testing.
"""

# The first model is an LSTM following by two dense layers
def create_LSTM(T,D, hidden_dim, output_dim,mid_dim=32):
    i = Input(shape=(T,D))
    x = LSTM(hidden_dim) (i)
    x = Dense(mid_dim,activation='sigmoid')(x)
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(i, x)
    return model

# The second model is an LSTM with global max pooling following by two dense layers
def create_LSTM_GMP(T,D,hidden_dim,output_dim,mid_dim=32):
    i = Input(shape=(T,D))
    x = LSTM(hidden_dim, return_sequences=True) (i)
    x = GlobalMaxPool1D()(x)
    x = Dense(mid_dim,activation='sigmoid')(x)
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(i, x)
    return model

# The third model is a GRU following by two dense layers
def create_GRU(T,D,hidden_dim,output_dim,mid_dim=32):
    i = Input(shape=(T,D))
    x = GRU(hidden_dim) (i)
    x = Dense(mid_dim,activation='sigmoid')(x)
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(i, x)
    return model

# The fourth model is a RNN following by two dense layers
def create_RNN(T,D,hidden_dim,output_dim,mid_dim=32):
    i = Input(shape=(T,D))
    x = SimpleRNN(hidden_dim) (i)
    x = Dense(mid_dim,activation='sigmoid')(x) 
    x = Dense(output_dim,activation='sigmoid')(x)
    model = Model(i, x)
    return model

# The fifth model is two CNNs following by two dense layers
def create_CNN(T,D, output_dim,mid_dim=32):
    i = Input(shape=(T,D))
    x = Conv1D(32, 3, 1,padding="same", activation='relu')(i)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3,1,padding="same", activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3,1,padding="same", activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(mid_dim,activation='relu')(x) 
    x = Dense(output_dim, activation='sigmoid')(x)
    model = Model(i, x)
    return model

# The sixth model is two CNNs following by LSTM and two dense layers
def create_CNN_LSTM(T,D,hidden_dim, output_dim,mid_dim=32):
    i = Input(shape=(T,D))
    x = Conv1D(32, 3, 1,padding="same", activation='relu')(i)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3,1,padding="same", activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3,1,padding="same", activation='relu')(x)
    x = LSTM(hidden_dim) (x)
    x = Dense(mid_dim,activation='relu')(x) 
    x = Dense(output_dim, activation='sigmoid')(x)
    model = Model(i, x)
    return model

# The seventh model is two CNNs following by GRU and two dense layers
def create_CNN_GRU(T,D, hidden_dim, output_dim,mid_dim=32):
    i = Input(shape=(T,D))
    x = Conv1D(32, 3, 1,padding="same", activation='relu')(i)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3,1,padding="same", activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3,1,padding="same", activation='relu')(x)
    x = GRU(hidden_dim) (x)
    x = Dense(mid_dim,activation='relu')(x) 
    x = Dense(output_dim, activation='sigmoid')(x)
    model = Model(i, x)
    return model

# The eighth model is a random forrest
def create_RF(n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators, criterion="gini",n_jobs=7)
    return model


stubFunctions = {
    'rf': create_RF,
    'rnn': create_RNN,
    'lstm': create_LSTM,
    'gru': create_GRU,
    'cnn': create_CNN,
    'lstm_gmp': create_LSTM_GMP,
    'cnn_lstm': create_CNN_LSTM,
    'cnn_gru':create_CNN_GRU
    }

# prediction_test calculate the performance metrics of a prediction model
def prediction_test(model,model_name,X_test, Y_test):
    print(model_name,'is validating ...')
    if model_name =='rf':
        outputs = model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1]))
    else:
        outputs = model(X_test)
    y_pred = (outputs > 0.5)
    cm = confusion_matrix(Y_test, y_pred)
    acc=(cm[0][0]+cm[1][1])/len(outputs)
    print("Test accuracy is: ",acc)
    print(classification_report(Y_test, y_pred))
    print("AUC: {}".format(roc_auc_score(Y_test, y_pred)))
    return roc_auc_score(Y_test, y_pred) 


"""
1 Invase
"""
landa = 0
# actor model is the selector for finding the most contributing features for each instance
def create_actor(input, output, first_dim):
    i = Input(shape=(input,))
    x = Dense(first_dim,activation='relu')(i)
    x = Dense(20,activation='relu')(x)
    x = Dense(30,activation='relu')(x)
    x = Dense(output, activation='sigmoid')(x)
    model = Model(i, x)
    return model

# custom loss for invase model, this loss using a reinforcement learning approach, articulates actor with KL divergeance 
# among the critic and the baseline
def my_loss(y_true, y_pred):
    d = y_pred.shape[1]
    sel_prob = y_true[:,:d]
    dis_prob = y_true[:,d:(d+1)]
    val_prob = y_true[:,(d+1):(d+2)]
    y_final = y_true[:,(d+2):]
    Reward1 = tf.reduce_sum(y_final * tf.math.log(dis_prob + 1e-8), axis = 1)
    Reward2 = tf.reduce_sum(y_final * tf.math.log(val_prob + 1e-8), axis = 1)
    Reward = Reward1 - Reward2
    loss1 = Reward * tf.reduce_sum( sel_prob * tf.math.log(y_pred + 1e-8)
                                   + (1-sel_prob) * tf.math.log(1-y_pred + 1e-8), axis = 1) - landa * tf.reduce_mean(y_pred, axis = 1)
    loss = tf.reduce_mean(-loss1)
    return loss

# Bernulli sampler
def Sample_M(gen_prob):
    n = gen_prob.shape[0]
    d = gen_prob.shape[1]
    samples = np.random.binomial(1, gen_prob, (n,d))
    return samples

# train invase model in which baseline is frozen and others will be trained
def train(X_train,y_train, batch_size, actor,critic,baseline,epoch_number,save_model,X_test,Y_test):
    global landa
    
    acc=-10
    iterations = int(np.round(X_train.shape[0]/batch_size))
    start=time.time()
    print('invase training gets started...')
    for epoch in range(epoch_number):
        for it in range(iterations):
            # Select a random batch of samples
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            x_batch = X_train[idx,:]
            y_batch = y_train[idx]
            gen_prob = actor.predict(x_batch) # m (number of original features) number of independent probabilities
            sel_prob = Sample_M(gen_prob) # a m size sparse vector
            dis_prob = critic.predict(x_batch*sel_prob[:, :, np.newaxis]) # critic's conditional probability distribution 
            d_loss = critic.train_on_batch(x_batch*sel_prob[:, :, np.newaxis], y_batch) # training critic with the sparse input
            val_prob = baseline.predict(x_batch)
            y_batch_final = np.concatenate( (sel_prob, np.asarray(dis_prob), np.asarray(val_prob), y_batch.reshape(-1,1)), axis = 1 )
            g_loss = actor.train_on_batch(x_batch, y_batch_final) # training the actor with the custom loss
            dialog = 'Epoch: ' + str(epoch) + ', d_loss (Acc)): ' + str(d_loss[1]) + ', g_loss: ' + str(np.round(g_loss,4))#+', v accuracy'+str(v_loss[1])
        print(dialog)
        new_acc = test(actor,critic,X_test,Y_test)
        if(new_acc>acc):
            acc = new_acc
            actor_weights = actor.get_weights()
            critic_weights = critic.get_weights()
    actor.set_weights(actor_weights)  
    critic.set_weights(critic_weights)
    end=time.time()
    print('End of training!!!')
    print('training time: ', end-start)
    return actor,critic

# validating invase during the training
def test(actor,critic,X_test,Y_test):
    gen_prob=actor(X_test)
    sel_prob = Sample_M(gen_prob)
    dis_prob = critic.predict(X_test*sel_prob[:, :, np.newaxis])
    dis_prob = (dis_prob > 0.5)
    cm = confusion_matrix(Y_test, dis_prob)
    acc=(cm[0][0]+cm[1][1])/len(dis_prob)
    print("Test accuracy is: ",acc)
    return acc

# test invase model
def test_explanation(actor,critic,X_test,Y_test):
    start=time.time()
    gen_prob=actor(X_test)
    sel_prob = Sample_M(gen_prob)
    dis_prob = critic.predict(X_test*sel_prob[:, :, np.newaxis])
    dis_prob = (dis_prob > 0.5)
    end=time.time()
    cm = confusion_matrix(Y_test, dis_prob)
    acc=(cm[0][0]+cm[1][1])/len(dis_prob)
    print("Test accuracy is: ",acc)
    print(classification_report(Y_test, dis_prob))
    print("AUC: {}".format(roc_auc_score(Y_test, dis_prob)))
    print('explanation time: ', end-start)
    res=np.hstack((X_test.reshape(X_test.shape[0],-1),sel_prob,dis_prob,Y_test.reshape(-1,1)))
    return res,roc_auc_score(Y_test, dis_prob)

"""
2 L2X
"""
# a custom Gumble softmax layer for L2X
class Gumble_Softmax(Layer):
    def __init__(self, tau0, k):
        super(Gumble_Softmax, self).__init__()
        self.tau0 = tau0
        self.k = k
 
   
    def get_config(self):
        config = super().get_config().copy()
        return config
    def __call__(self, logits):   
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)# [BATCH_SIZE, 1, d]
        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(shape =(batch_size, self.k, d), 
                                    minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval = 1.0)
        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis = 1) 
        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
        return K.in_train_phase(samples, discrete_logits)
    
    def compute_output_shape(self, input_shape):
        return input_shape

# this implementation converges around the baseline outputs
def L2X(X_train,y_train, batch_size,baseline,epoch_number,k,input_shape,X_test, Y_test,hidden_dim,mid_dim, output_dim,lr): 
    dis_prob=baseline(X_train)
    dis_prob = (dis_prob > 0.5)
    print('l2x training gets started...')
    start=time.time()
    # P(S|X) (actor)
    model_input = Input(shape=(input_shape,), dtype='float32')
    # increase a bit the number of neurons
    x = Dense(20,activation='relu')(model_input)
    x = Dense(20,activation='relu')(x)
    x = Dense(30,activation='relu')(x)
    logits = Dense(input_shape)(x)
    tau = 0.1
    samples = Gumble_Softmax(tau,k)(logits)
    # q(X_S) (Critique)
    new_model_input = Multiply()([model_input, samples])
    new_model_input=K.expand_dims(new_model_input, -1)
    x = LSTM(hidden_dim) (new_model_input)
    x = Dense(mid_dim,activation='relu')(x) # Adding mid level features
    preds = Dense(output_dim,activation='sigmoid')(x)
    X_train=X_train.reshape(-1,input_shape)
    X_test = X_test.reshape(-1,input_shape)
    model = Model(model_input, preds)
    adam = optimizers.Adam(lr = lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy']) 
    mc = ModelCheckpoint('best_l2x.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=0)
    callbacks_list = [mc]
    model.fit(X_train, dis_prob, callbacks = callbacks_list,validation_data = (X_test, Y_test), epochs=epoch_number, batch_size=batch_size)
    model = load_model('best_l2x.h5')
    pred_model = Model(model_input, samples)
    pred_model.compile(loss=None, optimizer='rmsprop', metrics=[None]) 
    end=time.time()
    print('End of training!!!')
    print('training time: ', end-start)
    return model, pred_model

def test_gumble(actor,critic,X_test,Y_test):
    start=time.time()
    gen_prob=actor(X_test)
    dis_prob = critic.predict(X_test)
    dis_prob = (dis_prob > 0.5)
    end=time.time()
    cm = confusion_matrix(Y_test, dis_prob)
    acc=(cm[0][0]+cm[1][1])/len(dis_prob)
    print("Test accuracy is: ",acc)
    print(classification_report(Y_test, dis_prob))
    print("AUC: {}".format(roc_auc_score(Y_test, dis_prob)))
    print('explanation time: ', end-start)
    res=np.hstack((X_test.reshape(X_test.shape[0],-1),gen_prob,dis_prob,Y_test.reshape(-1,1)))
    return res,roc_auc_score(Y_test, dis_prob)

"""
3 GSX
"""
# Gumble sigmoid layer for gsx
class Gumbel_Sigmoid(Layer):
    def __init__(self, units, tau0,kernel_regularizer=None, eps=1e-20):
        super(Gumbel_Sigmoid, self).__init__()
        self.tau0 = tau0
        self.eps = eps
        self.kernel_regularizer = kernel_regularizer
        self.units = units

    def build(self, input_shape): 
        self.kernel = self.add_weight(name = 'kernel', 
                                      shape = (input_shape[1], self.units), 
                                      initializer = 'normal', trainable = True, regularizer=self.kernel_regularizer) 
        super(Gumbel_Sigmoid, self).build(input_shape)
        
    def __call__(self,logits):
        """computes a gumbel sigmoid sample"""
        batch_size = tf.shape(logits)[0]
        d = tf.shape(logits)[1]

        uniform_1 = tf.random.uniform(shape =(batch_size, d), 
                                    minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval = 1.0)
        
        uniform_2 = tf.random.uniform(shape =(batch_size, d), 
                                    minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval = 1.0)
       
        gumbel = - K.log( K.log(uniform_2 + self.eps)/ K.log(uniform_1 + self.eps) +self.eps)
        samples = K.sigmoid((logits + gumbel) / self.tau0)
        return samples
     
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# this implementation converges around the baseline outputs
def GSX(X_train,y_train, batch_size,baseline,epoch_number,input_shape,X_test, Y_test,hidden_dim,mid_dim, output_dim,lr):
    global landa
    dis_prob = baseline(X_train)
    dis_prob = (dis_prob > 0.5)
    # P(S|X) (actor)
    print('gsx training gets started...')
    start=time.time()
    model_input = Input(shape=(input_shape,), dtype='float32')
    # increase a bit the number of neurons
    x = Dense(20,activation='relu')(model_input)
    x = Dense(20,activation='relu')(x)
    x = Dense(30,activation='relu')(x)
    logits = Dense(input_shape)(x)
    tau = 0.1
    samples = Gumbel_Sigmoid(input_shape,tau, l1(l=landa))(logits)
    # q(X_S) (Critique)
    new_model_input = Multiply()([model_input, samples])
    new_model_input=K.expand_dims(new_model_input, -1)
    x = LSTM(hidden_dim) (new_model_input)
    x = Dense(mid_dim,activation='relu')(x) # Adding mid level features
    preds = Dense(output_dim,activation='sigmoid')(x)
    X_train=X_train.reshape(-1,input_shape)
    X_test = X_test.reshape(-1,input_shape)
    model = Model(model_input, preds)
    adam = optimizers.Adam(lr = lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy']) 
    mc = ModelCheckpoint('best_gsx.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=0)
    callbacks_list = [mc]
    model.fit(X_train, dis_prob, callbacks = callbacks_list,validation_data = (X_test, Y_test), epochs=epoch_number, batch_size=batch_size)
    model = load_model('best_gsx.h5')
    pred_model = Model(model_input, samples)
    pred_model.compile(loss=None, optimizer='rmsprop', metrics=[None])
    end=time.time()
    print('End of training!!!')
    print('training time: ', end-start)
    return model, pred_model

def centroid_analysis(ds,k):
    wcss = []
    for i in range(1, k):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(ds)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, k), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def cluster_analysis(ds,k):
    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(ds)
    return np.round(kmeans.cluster_centers_), y_kmeans