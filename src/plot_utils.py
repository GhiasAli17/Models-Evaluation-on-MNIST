import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def display_confusion_matrix(cm, title):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = cm_normalized * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=range(10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical", values_format=".1f")
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=16)
    plt.show()

def plot_misclassified_images(misclassified_indices, x_test, y_test, model_pred, title):
    plt.figure(figsize=(15, 3))
    plt.suptitle(title, fontsize=16)
    for i, idx in enumerate(misclassified_indices[:10]):
        plt.subplot(1, 10, i+1)
        img = x_test[idx]
        if img.ndim == 3:
            img = img.squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f"T:{y_test[idx]} P:{model_pred[idx]}")
        plt.axis('off')
    plt.show()

def plot_training_graphs(history, model_name, title):
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)
    ax1.plot(epochs, history["accuracy"], 'bo-', label=f"{model_name} Training Accuracy")
    ax1.plot(epochs, history["val_accuracy"], 'b--', label=f"{model_name} Validation Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history["loss"], 'ro-', label=f"{model_name} Training Loss")
    ax2.plot(epochs, history["val_loss"], 'r--', label=f"{model_name} Validation Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()