from performanceTester import *

def train(model, test_loader, train_loader, classes, device, printPerformanceEveryNEpoch = -1, num_epochs = 100, learning_rate = 0.0001, weight_decay=0.01):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    train_loss_list = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}:', end=' ')
        train_loss = 0
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_list.append(train_loss / len(train_loader))
        print(f"Training loss = {train_loss_list[-1]}")

        if(printPerformanceEveryNEpoch != -1):
            if (epoch + 1) % 10 == 0:
                print("Performance Testing:")
                CalculatePerformanceMetrics(model, test_loader, classes, device, True)
                print("Performance Training:")
                CalculatePerformanceMetrics(model, train_loader, classes, device, True)
