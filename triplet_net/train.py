import torch
import torch.nn as nn

from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from backbone_net.backbone_model import Model


def main():
    # Define the data transform
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = 8
    
    # Load the training and test datasets
    train_dataset = datasets.ImageFolder('/home/munkacsiz/playroom/garbage/triplet_net/dataset_split/train', transform=transform) # TODO use relative path
    test_dataset = datasets.ImageFolder('/home/munkacsiz/playroom/garbage/triplet_net/dataset_split/test', transform=transform) # TODO use relative path
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model.load_state_dict(torch.load('/home/munkacsiz/playroom/garbage/triplet_net/backbone_net/checkpoints/backbone_net_epoch_25.pth')) # TODO use relative path
    model.fc1 = nn.Identity()
    model.to(device)
    
    # Define the metric learning components
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=10)
    
    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        test(train_dataset, test_dataset, model, accuracy_calculator)
    
    
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data) # 128-dimensional embeddings
        
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )
    
    
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    
    
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

if __name__ == "__main__":
    main()