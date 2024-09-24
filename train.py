import os
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Custom SSIM function used during training
def ssim(img1, img2, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2, count_include_pad=False)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2, count_include_pad=False)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2, count_include_pad=False) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2, count_include_pad=False) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2, count_include_pad=False) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# Custom dataset class (accepts 2 folder with images)
class CustomDataset(Dataset):
    def __init__(self, img_folder_A, img_folder_B, transform=None):
        self.img_folder_A = img_folder_A
        self.img_folder_B = img_folder_B
        self.transform = transform
        self.image_list_A = os.listdir(self.img_folder_A)
        self.image_list_B = os.listdir(self.img_folder_B)

    def __len__(self):
        return len(self.image_list_A)

    def __getitem__(self, idx):
        img_name_A = os.path.join(self.img_folder_A, self.image_list_A[idx])
        img_name_B = os.path.join(self.img_folder_B, self.image_list_B[idx])
        
        image_A = Image.open(img_name_A)
        image_B = Image.open(img_name_B)
        
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B

# CNN model
class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc_conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc_conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc_conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle
        self.middle_conv1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.middle_conv2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.middle_conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU())

        # Decoder
        self.dec_upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(nn.Conv2d(128 + 64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.dec_upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())

        self.dec_upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(nn.Conv2d(32 + 32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())

        # Additional upsample layer to ensure the output is 256x256
        self.upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1), nn.Sigmoid())

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(self.pool(enc1))
        enc3 = self.enc_conv3(self.pool(enc2))
        enc4 = self.enc_conv4(self.pool(enc3))

        # Middle
        middle = self.middle_conv3(self.middle_conv2(self.middle_conv1(self.pool(enc4))))

        # Decoder
        dec3 = self.dec_upconv3(middle)
        dec3 = torch.cat((dec3, enc4), dim=1)
        dec3 = self.dec_conv3(dec3)
        dec2 = self.dec_upconv2(dec3)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.dec_conv2(dec2)
        dec1 = self.dec_upconv1(dec2)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.dec_conv1(dec1)

        # Upsample to 256x256
        dec1 = self.upsample(dec1)
        out = self.final_conv(dec1)
        return out


# PSNR calculation
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def train_set(random_crop_val, dname, dataset_in, dataset_out):
    dname = dname + "_"
    print(dname+str(random_crop_val)+".log")
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(256, 256), scale=(random_crop_val, 1), antialias=True)
            ])
    train_dataset = CustomDataset(dataset_in, dataset_out, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=12)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeeperCNN()
    start_epoch = 1
    n_epochs = 16
    for i in range(n_epochs, start_epoch, -1):
        file = Path(dname+str(i)+".pth")
        if file.is_file():
            model.load_state_dict(torch.load(dname+str(i)+".pth"))
            print("Loaded: ", file.name)
            model.eval()
            start_epoch = i + 1
            break
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            ssim_loss = ssim(outputs, labels, window_size=11, size_average=True)
            loss = 1 - ssim_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            psnr = calculate_psnr(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
            epoch_psnr += psnr
            
            print(f"Epoch [{epoch}/{n_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, PSNR: {psnr:.4f}")
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)

        print(f"Epoch [{epoch}], Avg Loss: {avg_epoch_loss:.4f}, Avg PSNR: {avg_epoch_psnr:.4f}")
        with open(dname+str(random_crop_val)+".log", "a") as f:
            f.write(f"Epoch [{epoch}], Avg Loss: {avg_epoch_loss:.4f}, Avg PSNR: {avg_epoch_psnr:.4f}\n")
            torch.save(model.state_dict(), dname+str(epoch)+".pth")

if __name__ == '__main__':
    # No RC — value = 1
    train_set(1, "models/iso40000-07-10-r-balanced", f"noised/all", "clean/all")
