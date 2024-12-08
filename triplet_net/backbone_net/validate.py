
from torchvision import datasets, transforms
import torch.nn as nn
import torch

from backbone_model import Model

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

valid_dataset = datasets.ImageFolder('/home/munkacsiz/playroom/garbage/backbone_net/reduced_dataset_split/val', transform=transform)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=False)


model = Model()

loss_fn = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(num_classes=20)
model.to(device)

model.load_state_dict(torch.load('/home/munkacsiz/playroom/garbage/backbone_net/checkpoints/backbone_net_epoch_25.pth'))

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
print(f"Accuracy: {100 * correct/total:.2f}%")
