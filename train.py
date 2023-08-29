import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Function for calculating PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    
# Define the CNN Model
class ColorReductionNet(nn.Module):
    def __init__(self):
        super(ColorReductionNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset_A = datasets.ImageFolder(root='./test_images_256', transform=transform)
dataset_B = datasets.ImageFolder(root='./test_images_256_noised', transform=transform)

loader_A = DataLoader(dataset_A, batch_size=16, shuffle=False)
loader_B = DataLoader(dataset_B, batch_size=16, shuffle=False)

# Put the model on GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = ColorReductionNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
# Train the model
for epoch in range(num_epochs):
    epoch_avg_psnr = 0
    epoch_avg_ssim = 0
    sample_count = 0
    
    for batch_idx, ((data_A, _), (data_B, _)) in enumerate(zip(loader_A, loader_B)):
        data_A = data_A.to(device)
        data_B = data_B.to(device)
        
        optimizer.zero_grad()
        outputs = model(data_A)
        loss = criterion(outputs, data_B)
        
        loss.backward()
        optimizer.step()
        
        # Convert tensors to numpy arrays for PSNR and SSIM calculations
        data_A_np = data_A.cpu().detach().numpy().transpose((0, 2, 3, 1))
        outputs_np = outputs.cpu().detach().numpy().transpose((0, 2, 3, 1))
        
        for i in range(data_A_np.shape[0]):
            data_scaled = (data_A_np[i] * 255).astype(np.uint8)
            output_scaled = (outputs_np[i] * 255).astype(np.uint8)
            
            cur_psnr = calculate_psnr(data_scaled, output_scaled)
            cur_ssim = ssim(data_scaled, output_scaled, multichannel=True)
            
            epoch_avg_psnr += cur_psnr
            epoch_avg_ssim += cur_ssim
            sample_count += 1
        
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader_A)}, Loss: {loss.item():.4f}")
    
    epoch_avg_psnr /= sample_count
    epoch_avg_ssim /= sample_count
    
    print(f"After Epoch {epoch+1}, Average PSNR: {epoch_avg_psnr:.4f}, Average SSIM: {epoch_avg_ssim:.4f}")
    
    with open('train.log', 'a') as f:
        f.write(f"\nEpoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Average PSNR: {epoch_avg_psnr}, Average SSIM: {epoch_avg_ssim}")
    
    # Save model for the current epoch
    torch.save(model.state_dict(), f"color_reduction_net_epoch_{epoch+1}.pth")

# Test the model
model.eval()
with torch.no_grad():
    for data, _ in loader_A:
        data = data.to(device)
        output = model(data)
