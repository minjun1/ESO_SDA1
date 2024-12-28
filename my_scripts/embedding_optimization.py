"""
embedding_optimization.py

This script implements fine-tuning of a learned embedding space to separate 
two classes (events vs. noise) more distinctly. The approach:

1. Load a pre-trained CNN that has an 'embedding_layer'.
2. Produce embeddings for the training data, then compute the class centroid for 
   "noise" and "event" samples.
3. Define a custom distance-based loss function that pulls each sample closer 
   to its class centroid in the embedding space.
4. Fine-tune the embedding model to optimize (reduce) that distance-based loss.
5. Use a distance threshold for classification: If sample embedding is 
   closer to event centroid (beyond some margin from noise centroid) => event, 
   otherwise => noise.
6. Evaluate performance with confusion matrix, classification report, 
   PR/ROC curves, etc., focusing on imbalanced data metrics.

Dependencies:
- TensorFlow / Keras
- NumPy
- SciPy / scikit-learn for metrics
- Matplotlib / Seaborn for plots

Author: [Your Name]
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, roc_auc_score, 
                             precision_recall_curve, auc)

# -------------------------------------------------------------------
# 1. Distance Loss for Fine-Tuning
# -------------------------------------------------------------------

def build_distance_loss(event_center, noise_center, distance_margin=10.0):
    """
    Creates a custom loss function that pulls embeddings toward 
    their respective class centroid. Each sample is labeled as 0 (noise) or 1 (event).

    distance_margin : float
        A constant that can be added to the loss if desired to control scaling.
    """
    # Convert centers to constant tensors
    event_center_tensor = tf.constant(event_center, dtype=tf.float32)
    noise_center_tensor = tf.constant(noise_center, dtype=tf.float32)

    def custom_loss(y_true, y_pred):
        """
        y_true: shape (N,) with 0 or 1
        y_pred: shape (N, embedding_dim) (the embedding)
        """
        # Convert y_true to float and create masks
        y_true = tf.cast(tf.squeeze(y_true), tf.float32)
        event_mask = tf.cast(tf.equal(y_true, 1), tf.float32)
        noise_mask = tf.cast(tf.equal(y_true, 0), tf.float32)

        # Distances to each class centroid
        # Euclidean distance = sum of squared differences across embedding dims
        event_distances = tf.reduce_sum(tf.square(y_pred - event_center_tensor), axis=1)
        noise_distances = tf.reduce_sum(tf.square(y_pred - noise_center_tensor), axis=1)

        # Weighted sum of distances
        # Only add event_distances for the event samples, noise_distances for noise
        sum_event_dist = tf.reduce_sum(event_mask * event_distances)
        sum_noise_dist = tf.reduce_sum(noise_mask * noise_distances)

        # Normalization factor = total number of samples
        # or sum of (event_mask + noise_mask) just in case we want partial subsets
        norm_factor = tf.reduce_sum(event_mask) + tf.reduce_sum(noise_mask)
        norm_factor = tf.maximum(norm_factor, 1.0)  # avoid division by zero

        total_loss = (sum_event_dist + sum_noise_dist) / norm_factor

        # Optionally add distance_margin for extra scaling
        return total_loss + distance_margin

    return custom_loss


# -------------------------------------------------------------------
# 2. Utility Functions for Embedding and Centroids
# -------------------------------------------------------------------

def compute_embeddings(model, X_data, batch_size=1):
    """
    Produce embeddings for data X_data using a Keras model's output. 
    The model should directly output embeddings (e.g. 'embedding_layer').

    Returns:
    --------
    embeddings : np.ndarray of shape (N, embedding_dim)
    """
    return model.predict(X_data, batch_size=batch_size)


def compute_class_centers(embeddings, labels):
    """
    Compute the mean embedding (centroid) for two classes: 0 (noise), 1 (event).

    embeddings : np.ndarray, shape (N, embedding_dim)
    labels     : np.ndarray, shape (N,), containing 0 or 1

    Returns:
    --------
    event_center : np.ndarray, shape (embedding_dim,)
    noise_center : np.ndarray, shape (embedding_dim,)
    """
    event_embeddings = embeddings[labels == 1]
    noise_embeddings = embeddings[labels == 0]

    event_center = np.mean(event_embeddings, axis=0)
    noise_center = np.mean(noise_embeddings, axis=0)

    return event_center, noise_center


def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two 1D vectors."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


# -------------------------------------------------------------------
# 3. Fine-Tuning the Embedding Model
# -------------------------------------------------------------------

def fine_tune_embedding_model(
    base_model_path,
    X_train,
    y_train,
    event_center,
    noise_center,
    distance_margin=10.0,
    batch_size=64,
    epochs=500,
    checkpoint_path='embedding_checkpoints/model_epoch_{epoch:03d}.h5'
):
    """
    Loads a pre-trained model from base_model_path, extracts the embedding_layer,
    and fine-tunes it by minimizing the distance-based custom loss that 
    pulls samples to their respective class centroid.

    Returns:
    --------
    embedding_model : tf.keras.Model
        The fine-tuned embedding model.
    history : tf.keras.callbacks.History
        Keras training history.
    """

    # Multi-GPU strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Load entire pre-trained model
        base_model = load_model(base_model_path)
        # Create embedding model that outputs just the embedding_layer
        embedding_model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('embedding_layer').output
        )
        
        # Setup optimizer & schedule
        initial_learning_rate = 0.001
        lr_schedule = ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)

        # Compile with our custom distance loss
        dist_loss = build_distance_loss(event_center, noise_center, distance_margin)
        embedding_model.compile(optimizer=optimizer, loss=dist_loss)

        # Setup checkpoint callback
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            save_freq='epoch',
            verbose=1,
            save_best_only=False,  # We can save every epoch or choose a different strategy
            save_weights_only=True
        )

        # Train
        history = embedding_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint]
        )

    return embedding_model, history


# -------------------------------------------------------------------
# 4. Distance-Based Classification & Evaluation
# -------------------------------------------------------------------

def classify_by_distance(embeddings, event_center, noise_center, threshold=None):
    """
    Given embeddings, classify them as event/noise based on distance 
    to the 'noise_center' or 'event_center' plus a threshold.

    If threshold is None, we might use the minimal-distance rule:
       label = 1 if dist_to_event < dist_to_noise else 0
    Otherwise, you might do something like:
       label = 1 if dist_to_noise > threshold else 0
    """
    distances_to_noise = np.array([euclidean_distance(e, noise_center) for e in embeddings])
    distances_to_event = np.array([euclidean_distance(e, event_center) for e in embeddings])

    if threshold is None:
        # Classification by whichever center is closer
        predictions = np.where(distances_to_event < distances_to_noise, 1, 0)
    else:
        # E.g. if the distance to noise is > threshold => classify as event
        predictions = np.where(distances_to_noise > threshold, 1, 0)

    return predictions, distances_to_event, distances_to_noise


def evaluate_performance(y_true, y_pred, title="Evaluation"):
    """
    Print confusion matrix, classification report, and basic metrics 
    (accuracy, precision, recall, F1).
    """
    conf_m = confusion_matrix(y_true, y_pred)
    print(f"--- {title} ---")
    print("Confusion Matrix:\n", conf_m)
    
    # Show it as a heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_m, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title} - Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Noise','Event']))

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")


def plot_pr_curve(y_true, distances, label_name='Distance-based'):
    """
    Example PR-curve if we treat 'distances' as the score for class=1 (event).
    Usually, for 'event' = 1, you want smaller distance => more confident event.
    This can be inverted or you can do e.g. 1 - distance if you want a "probability-like" measure.

    Implementation detail: If you want the "score" to be higher for event-likely, 
    you might do score = -distance or (max-dist). 
    For demonstration, we'll use negative distance so that a smaller distance 
    => bigger "score" for class=1.
    """
    # Convert distances so that smaller => higher "score"
    # This is a common trick when using metrics that expect 
    # a bigger score for more-likely positives.
    event_score = -distances  # invert

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, event_score)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(6,5))
    plt.plot(recall_vals, precision_vals, label=f"{label_name} (PR-AUC={pr_auc:.2f})")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.show()

    return pr_auc


def plot_roc_curve(y_true, distances, label_name='Distance-based'):
    """
    Plot ROC curve, similarly turning distance into a "score" 
    where smaller distance => more positive (event).
    """
    score = -distances
    fpr, tpr, _ = roc_curve(y_true, score)
    roc_auc = roc_auc_score(y_true, score)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'{label_name} (AUC={roc_auc:.2f})')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc
