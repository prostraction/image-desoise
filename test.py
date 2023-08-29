import torch
from torch import nn
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image
from pytorch_msssim import ssim
import os

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


# Initialize the model and load trained weights
model = ColorReductionNet()
model.load_state_dict(torch.load("color_reduction_net_epoch_3.pth"))  # replace with your model filename
model.eval()

# Put the model on GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_dataset = datasets.ImageFolder(root='./test_images_256_noised', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Set batch_size to 1 for simplicity

# Process and save the outputs
output_dir = "denoised"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with torch.no_grad():
    for idx, (data, _) in enumerate(test_loader):
        data = data.to(device)
        output = model(data)
        
        # Get original filename
        image_path = test_dataset.imgs[idx][0]
        original_filename = os.path.basename(image_path)
        
        # Convert tensor to PIL image and save with original filename
        output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_image.save(os.path.join(output_dir, original_filename))