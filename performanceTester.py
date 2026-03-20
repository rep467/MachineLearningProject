import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def CalculateAccuracy(model, test_loader, device='CPU', printData = False):
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            y_true = labels.to(device)
            outputs = model(images)
            _, y_pred = torch.max(outputs.data, 1)
            test_acc += (y_pred == y_true).sum().item()

    accuracy = 100 * test_acc / len(test_loader.dataset)
    if(printData):
        print(f"Test set accuracy = {accuracy} %")
    
    return accuracy

def CalculatePerformanceMetrics(model, test_loader, classes, device, prinData = False):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    result = classification_report(all_labels, all_preds, target_names=classes)

    if(prinData):
        print(result)
    
    return classification_report(all_labels, all_preds, target_names=classes, output_dict=True)