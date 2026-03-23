from performanceTester import *

def train(model, test_loader, train_loader, classes, device, printPerformanceEveryNEpoch = -1, num_epochs = 100, learning_rate = 0.0001, weight_decay=0.01, earlyStop=False, testAgainstTrainingSet=False, classs_weights=None):
    criterion = torch.nn.CrossEntropyLoss(weight=classs_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    train_loss_list = []
    bestScore = -1
    bestScoreEpochNr = -1

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


        result = CalculatePerformanceMetrics(model, test_loader, classes, device, False)

        currentScore = result['weighted avg']['f1-score']
        if currentScore > bestScore:
            bestScore = currentScore
            bestScoreEpochNr = epoch
            torch.save(model.state_dict(), 'best_state.pth')
        elif earlyStop:
            if epoch - bestScoreEpochNr > 5:
                print(f"Early access triggered at {epoch+1}")
                break


        if(printPerformanceEveryNEpoch != -1):
            if (epoch + 1) % printPerformanceEveryNEpoch == 0:
                print("Performance Validation:")
                CalculatePerformanceMetrics(model, test_loader, classes, device, True)
                if testAgainstTrainingSet:
                    print("Performance Training:")
                    CalculatePerformanceMetrics(model, train_loader, classes, device, True)

    if bestScoreEpochNr != num_epochs -1:
       model.load_state_dict(torch.load('best_state.pth')) 
