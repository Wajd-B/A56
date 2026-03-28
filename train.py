import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from architecture import tinyimg_CNET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.ToTensor()])

#datasett
trainset = datasets.ImageFolder(
    root=r"C:\Users\asus\Desktop\AI417_Datasets\tiny-imagenet-200\train",
    transform=transform
)

#dataLoader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

#model
model = tinyimg_CNET().to(device)

# loss 
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

torch.save(model.state_dict(), "tinyimg_model.pth")