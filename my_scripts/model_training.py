"""
model_training.py

This script covers:
1) Loading data and initial visualization
2) Building a CNN (with multiple Conv2D, pooling, and fully-connected layers)
3) Training with an appropriate optimizer (Adam with an exponential decay schedule)
4) Handling imbalanced data: displaying relevant metrics (precision, recall, F1, confusion matrix, ROC/PR curves, etc.)
5) Generating embedding vectors and t-SNE visualization to inspect learned representations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization, LeakyReLU)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, roc_curve, auc,
                             matthews_corrcoef, balanced_accuracy_score)
from sklearn.manifold import TSNE


def visualize_examples(data_array, num_examples=6):
    """
    Visualize random event/noise patches from a 4D array of shape (N, H, W, 1).

    Parameters
    ----------
    data_array : np.ndarray
        The data array containing your images, shape (N, H, W, 1).
    num_examples : int
        Number of examples to visualize (will be arranged in a grid).
    """
    if len(data_array) == 0:
        print("No data to visualize.")
        return
    
    # Randomly select indices
    random_indices = np.random.choice(data_array.shape[0], num_examples, replace=False)
    rows = num_examples // 2 if (num_examples % 2 == 0) else (num_examples // 2 + 1)
    fig, axs = plt.subplots(rows, 2, figsize=(15, rows * 4))

    # If num_examples < 2, 'axs' won't be a 2D array
    axs = np.atleast_2d(axs)

    for idx, ax_ in enumerate(axs.flatten()):
        if idx >= num_examples:
            ax_.axis('off')
            continue
        img_index = random_indices[idx]
        # data_array[img_index, :, :, 0] -> 2D slice
        im = ax_.imshow(data_array[img_index, :, :, 0], cmap='gray', aspect='auto')
        ax_.set_title(f"Sample Index: {img_index}")
        ax_.axis('on')
    plt.tight_layout()
    plt.show()


def create_advanced_cnn(input_shape):
    """
    Build a deeper CNN architecture with multiple Conv2D/Pooling layers
    and L2 regularization. The final dense layer (with 'sigmoid') 
    outputs a binary classification prediction.
    """
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001), input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(512, kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(512, kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        # Embedding layer to extract intermediate representations
        Dense(128, kernel_regularizer=l2(0.001), name='embedding_layer'),
        LeakyReLU(alpha=0.1),
        Dense(1, activation='sigmoid')  # For binary classification
    ])
    
    return model


def train_cnn_model(
    X_train, y_train,
    save_path='saved_models/model.h5',
    batch_size=32,
    epochs=20,
    val_split=0.1,
    initial_lr=1e-4
):
    """
    Train the CNN model using MirroredStrategy (multi-GPU if available),
    with an exponential learning rate decay schedule.
    Saves the best model by 'val_accuracy'.
    """
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_shape = X_train.shape[1:]  # e.g. (288, 695, 1)
        model = create_advanced_cnn(input_shape)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    # Define a checkpoint callback to save the best model
    checkpoint = ModelCheckpoint(
        filepath=save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=False
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=val_split,
        batch_size=batch_size * strategy.num_replicas_in_sync,
        verbose=2,
        shuffle=True,
        callbacks=[checkpoint]
    )
    
    print("Training completed. Best model is saved to:", save_path)
    return model, history


def plot_history(history, out_path=None):
    """
    Plot training and validation accuracy/loss from the history object.
    Optionally save the plots to a file.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Training curves saved to {out_path}")
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data, produce confusion matrix, classification report,
    and compute standard metrics: accuracy, precision, recall, F1-score.

    Focus on imbalanced data concerns.
    """
    # Predictions
    y_pred_prob = model.predict(X_test, batch_size=1)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    print("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Noise','Event'])
    print(report)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (focus on false positives for imbalanced data)")
    print(f"Recall:    {rec:.4f} (focus on false negatives for imbalanced data)")
    print(f"F1 Score:  {f1:.4f} (harmonic mean of precision and recall)")

    return y_pred


def plot_pr_curve(y_test, y_score, label_name='Model'):
    """
    Plot Precision-Recall (PR) curve for an imbalanced dataset.

    - Precision: proportion of positive identifications that were actually correct
    - Recall: proportion of actual positives that were identified correctly
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label=f'{label_name} (AUC={pr_auc:.2f})')
    plt.title("Precision-Recall (PR) Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"PR-AUC = {pr_auc:.3f}")
    return pr_auc


def plot_roc_curve(y_test, y_score, label_name='Model'):
    """
    Plot ROC curve. For an imbalanced dataset, ROC can be misleading (if negative class is large),
    but it's still a common metric.
    """
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{label_name} (AUC={roc_auc:.2f})')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"ROC-AUC = {roc_auc:.3f}")
    return roc_auc


def generate_embeddings(model, X_data):
    """
    Extract embeddings from the 'embedding_layer' of the model.
    You can visualize these via t-SNE or another dimensionality reduction method.
    """
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('embedding_layer').output
    )
    embeddings = embedding_model.predict(X_data, batch_size=1)
    return embeddings


def plot_tsne(embeddings, labels, title="t-SNE Visualization"):
    """
    Apply t-SNE to the embeddings and plot the 2D scatter.
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(7,6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Class (0=Noise, 1=Event)')
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.show()
