
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)



# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
# model = nn.Sequential(
#     nn.Conv2d(1, 5, 3),
#     nn.ReLU(),
#     nn.Linear(1728, 10)
#     nn.ReLU(),
#     nn.Conv2d(5, 3, 3),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Dropout(p=0.2)
#     nn.Linear(1728, 10)
#     nn.ReLU(),
# ).to(device)
model = nn.Sequential(
    nn.Conv2d(1, 5, 5),
    nn.ReLU(),
    nn.Conv2d(5, 3, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(1728, 50)
    nn.ReLU(),
    nn.Linear(50, 10)
    nn.ReLU(),
    nn.Linear(10, 10)
).to(device)
print(model)
train_loss_ls = []
test_loss_ls = []


batch_size = 100
learning_rate = .01

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    test_loss, correct = test(test_dataloader, model, loss_fn)
    train_loss_ls += [float(train_loss)]
    test_loss_ls += [test_loss]

print("Done!")

pyplot.figure()
_ = pyplot.plot(train_loss_ls)
_ = pyplot.plot(test_loss_ls)
_ = pyplot.legend(['Training Loss', 'Test Loss'])
pyplot.show()



# torch.save(model.state_dict(), "conv_model.pth")
# print("Saved PyTorch Model State to conv_model.pth")

# model = nn.Sequential(
#     nn.Conv2d(1, 5, 3),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(3380, 10)
# ).to(device)
# model.load_state_dict(torch.load("conv_model.pth"))



classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
idx = 2
x, y = test_data[idx][0], test_data[idx][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

pyplot.figure()
pyplot.imshow(x[0], cmap='gray')
pyplot.show()




model.eval()
idx = 2
x, y = test_data[idx][0], test_data[idx][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


test_results = []
for test_data_point in test_data:
    x, y = test_data_point[0], test_data_point[1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        test_results += [(predicted, actual, predicted == actual)]


sum([b for _, __, b in test_results]) / len(test_results)
