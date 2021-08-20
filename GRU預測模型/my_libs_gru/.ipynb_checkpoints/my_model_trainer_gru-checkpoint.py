import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow import keras


### 模型訓練(GRU)

##### 訓練集&測試集

def train_test_split(df):
    test_samples = int(df.shape[0]*0.2)         # 82拆分
    
    # 訓練集
    train_data = df.iloc[:-test_samples, :]
    train_set = train_data.iloc[:, 1:].values   # 取得train_set(array)
 
    # 測試集
    test_data = df.iloc[-test_samples:, :]
    test_set = test_data.iloc[:, 1:].values     # 取得test_set(array)
       
    return train_set, test_set

##### 資料標準化

def data_normalization(input_set):
    
    sc = StandardScaler()
    input_set_sc = sc.fit_transform(input_set[:, :])

    sc_target = StandardScaler()
    sc_target.fit_transform(input_set[:, 0:1])
    
    return input_set_sc, sc_target

##### 創造X、y資料

def split_Xy(input_set_sc, n_past, n_future):
    X = []
    y = []
    for i in range(n_past, len(input_set_sc)-n_future+1):   
        X.append(input_set_sc[i-n_past:i, :])               
        y.append(input_set_sc[i:i+n_future, 0])             
        
    X, y = np.array(X), np.array(y) 
    
    print("X's shape: {}".format(X.shape))
    print("y's shape: {}".format(y.shape))
    
    return X, y

##### 建立模型

def model_creator(n_past, n_features, output):
    model = Sequential()
    model.add(GRU(units=100, activation="tanh", 
                   input_shape=(n_past, n_features), 
                   return_sequences=True)) 
    model.add(GRU(units=50, activation="tanh", return_sequences=False))          
    model.add(Dropout(0.2))
    model.add(Dense(output))
    model.summary() 
     
    return model

##### 訓練模型

def model_trainer(model, X_train, y_train, epoch, batch_size, loss_visualize=False):
    
    model.compile(optimizer="adam", 
                  loss="mean_squared_error")
    
    # 提升訓練效率
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, 
                          save_best_only=True, save_weights_only=True)
    
    history = model.fit(X_train, y_train, epochs=epoch, 
                        batch_size=batch_size,
                        callbacks=[es, rlr, mcp], 
                        validation_split=0.2,
                        shuffle=False,
                        verbose=1)
    
    if loss_visualize == True:  # 訓練過程以圖表顯示
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
    
    return model


##### 驗證模型準確率

def model_validation(model, X, y, sc_target):
    
    # 預測
    prediction = model.predict(X)
    prediction = sc_target.inverse_transform(prediction)
    actual = sc_target.inverse_transform(y)
    
    rmse_result = []
    for i in range(len(prediction)):
        rmse = mean_squared_error(prediction[i], actual[i], squared=False)
        rmse_result.append(rmse)

    pd.DataFrame({"rmse":rmse_result}).plot()
    
    return prediction, actual


    