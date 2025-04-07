import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from lenet5 import QuantLeNet5

# prepare model
model = QuantLeNet5()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
model = torch.ao.quantization.prepare_qat(model.train())

# train setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# qat train loop
num_iters = 50
for epoch in range(num_iters):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_iters} | '
                  f'Batch: {batch}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f}')


# eval
model.eval()
torch.save(model.state_dict(), "fake_quant_fp_lenet_weight.pth")

model = torch.ao.quantization.convert(model)
# trace
trace_data = torch.randn(1,1,28,28)
traced_model = torch.jit.trace(model, trace_data)
traced_model.save("quant_lenet.pt")

print("saved")
