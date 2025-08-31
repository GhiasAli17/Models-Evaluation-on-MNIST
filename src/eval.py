import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torchsummary import summary

def evaluate_model(model, data_loader, is_cnn=False):
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    if is_cnn:
        model_summary = summary(model, input_size=(1, 28, 28))
    else:
        model_summary = summary(model, input_size=(784,))
        
    return acc, report, cm, np.array(all_preds), np.array(all_labels),model_summary