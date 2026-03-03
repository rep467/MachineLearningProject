import torch

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