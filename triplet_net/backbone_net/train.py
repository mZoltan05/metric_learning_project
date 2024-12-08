from torchvision import datasets, transforms
import torch.nn as nn
import torch

from backbone_model import Model

def main():
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder('/home/munkacsiz/playroom/garbage/backbone_net/reduced_dataset_split/train', transform=transform) # TODO use relative path
    test_dataset = datasets.ImageFolder('/home/munkacsiz/playroom/garbage/backbone_net/reduced_dataset_split/test', transform=transform) # TODO use relative path

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


    model = Model()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(100):
        train(model, loss_fn, device, train_loader, optimizer, epoch)
        test(model, loss_fn, device, test_loader)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/backbone_net_epoch_{epoch}.pth')


def train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")


def test(model, loss_fn, device, test_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
    print(f"Validation Loss: {val_loss/len(test_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

if __name__ == '__main__':
    main()
    