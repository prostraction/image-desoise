import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import os

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



if __name__ == '__main__':
    file_list = os.listdir("models/")
    model_files = [file for file in file_list if file.endswith(".pth")]

    transform = transforms.Compose([transforms.ToTensor()])
    for model_file in model_files:
        if model_file == "real_noise_15.pth":
            model_name = os.path.splitext(model_file)[0]
            output_dir = os.path.join("outputs", model_name)
            os.makedirs(output_dir, exist_ok=True)

            model = DeeperCNN()
            model.load_state_dict(torch.load("models/" + model_file))
            model.eval()

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            transform = transforms.Compose([transforms.ToTensor()])
            test_dataset = datasets.ImageFolder(root='test/', transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

            with torch.no_grad():
                for idx, (data, _) in enumerate(test_loader):
                    data = data.to(device)
                    output = model(data)
                    for i in range(output.shape[0]):
                        image_path = test_dataset.imgs[idx * output.shape[0] + i][0] 
                        original_filename = os.path.basename(image_path)

                        output_image = transforms.ToPILImage()(output[i].cpu())
                        output_image.save(os.path.join(output_dir, f"{original_filename}_{model_name}.jpg"))