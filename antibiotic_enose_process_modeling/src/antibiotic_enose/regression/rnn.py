from __future__ import annotations
import numpy as np
from ..preprocessing import make_regression_sequences
from ..metrics import regression_metrics


def build_rnn_regressor(kind, input_shape, units=(100,50), learning_rate=1e-3,
                        activation="relu"):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense
    kind = kind.upper()
    model=Sequential()
    if kind == "LSTM":
        model.add(LSTM(units[0], activation=activation, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units[1], activation=activation))
    elif kind == "BILSTM":
        model.add(Bidirectional(LSTM(units[0], activation=activation, return_sequences=True), input_shape=input_shape))
        model.add(Bidirectional(LSTM(units[1], activation=activation)))
    elif kind == "BIGRU":
        model.add(Bidirectional(GRU(units[0], activation=activation, return_sequences=True), input_shape=input_shape))
        model.add(Bidirectional(GRU(units[1], activation=activation)))
    else:
        raise ValueError("kind must be LSTM, BiLSTM, or BiGRU")
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def fit_rnn_regressor(train_df, test_df, target, kind, window=6,
                      units=(100,50), learning_rate=1e-3, epochs=300,
                      batch_size=32, validation_split=0.15, verbose=0):
    Xtr,ytr,_,_ = make_regression_sequences(train_df,target,window)
    Xte,yte,_,_ = make_regression_sequences(test_df,target,window)
    mu=Xtr.mean(axis=(0,1),keepdims=True)
    sd=Xtr.std(axis=(0,1),keepdims=True)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    model=build_rnn_regressor(kind,Xtr.shape[1:],units,learning_rate)
    hist=model.fit(Xtr,ytr,validation_split=validation_split,epochs=epochs,
                   batch_size=batch_size,verbose=verbose)
    pred=model.predict(Xte,verbose=0).ravel()
    return model,hist.history,yte,pred,regression_metrics(yte,pred),(mu,sd)
