from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from ..metrics import classification_metrics


def build_lstm_classifier(input_shape, n_classes=3, units=(100, 50), learning_rate=1e-3):
    """Configurable LSTM comparator.

    The exact historical classification-LSTM architecture was not fully preserved;
    therefore unit counts and training hyperparameters are exposed explicitly.
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        LSTM(units[1]),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_lstm_classifier(X, y, test_size=0.30, seed=20260913, units=(100, 50),
                        learning_rate=1e-3, epochs=200, batch_size=32, verbose=0):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    mu = X[tr].mean(axis=(0,1), keepdims=True)
    sd = X[tr].std(axis=(0,1), keepdims=True) + 1e-8
    Xn = (X-mu)/sd
    model = build_lstm_classifier(X.shape[1:], len(np.unique(y)), units, learning_rate)
    hist = model.fit(Xn[tr], y[tr], validation_split=0.15, epochs=epochs,
                     batch_size=batch_size, verbose=verbose)
    pred = model.predict(Xn[te], verbose=0).argmax(axis=1)
    return model, hist.history, classification_metrics(y[te], pred), (te, pred), (mu, sd)
