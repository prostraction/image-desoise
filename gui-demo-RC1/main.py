import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from torch import nn
from torchvision import transforms
import traceback

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

# Preprocessing
transform = transforms.Compose([transforms.ToTensor()])

# Initialize the model
model = DeeperCNN()

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1280x720")
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.left_panel = tk.Frame(root, width=200, height=720, bg="#f0f0f0", padx=10, pady=10)
        self.left_panel.grid(row=0, column=0, sticky="ns")

        self.right_panel = tk.Frame(root, bg="#ffffff")
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        
        self.label = tk.Label(self.left_panel, text="Choose model", bg="#f0f0f0", font=("Helvetica", 12))
        self.label.pack(pady=10)
        
        self.model_var = tk.StringVar(value="CNN MSE ISO 40000")
        self.model_option = ttk.Combobox(self.left_panel, textvariable=self.model_var, state="readonly", font=("Helvetica", 10))
        self.model_option['values'] = ("CNN MSE ISO 40000", "CNN SSIM", "CNN Perlin", "CNN Wavelet", "CNN Gauss")
        self.model_option.pack(pady=10, fill="x")
        
        self.choose_image_btn = tk.Button(self.left_panel, text="Choose Image", command=self.choose_image, font=("Helvetica", 10))
        self.choose_image_btn.pack(pady=10, fill="x")
        
        self.process_btn = tk.Button(self.left_panel, text="Process", command=self.process_image, font=("Helvetica", 10))
        self.process_btn.pack(pady=10, fill="x")
        
        self.image_label = tk.Label(self.right_panel, bg="#ffffff")
        self.image_label.pack(expand=True)
        
        self.input_image = None
        self.processed_image = None

        save_button = tk.Button(self.left_panel, text="Save", command=self.save_image, font=("Helvetica", 10))
        save_button.pack(pady=10, fill="x")
        
        self.root.bind('<Configure>', self.on_resize)
    
    def load_model(self):
        model_file_map = {
            "CNN MSE ISO 40000": "CNNMSE40000ISO.pth",
            "CNN SSIM": "CNNSSIM.pth",
            "CNN Perlin": "CNNPerlin.pth",
            "CNN Wavelet": "CNNWavelet.pth",
            "CNN Gauss": "CNNGauss.pth"
        }
        model_file = model_file_map.get(self.model_var.get())
        if model_file:
            try:
                model.load_state_dict(torch.load(model_file))
                model.eval()
                print(f"Loaded model: {model_file}")
            except Exception as e:
                print(f"Error loading model {model_file}: {e}")
                traceback.print_exc()
        else:
            print("Invalid model selection")

    def choose_image(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
            if file_path:
                self.input_image = Image.open(file_path)
                self.processed_image = None
                self.display_image(self.input_image)
        except Exception as e:
            print("Error choosing image:", e)
            traceback.print_exc()
    
    def display_image(self, image):
        try:
            max_width = self.right_panel.winfo_width()
            max_height = self.right_panel.winfo_height()
            width, height = image.size
            aspect_ratio = width / height
            if width > max_width or height > max_height:
                if aspect_ratio > 1:
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)
                image = image.resize((new_width, new_height))
            img = ImageTk.PhotoImage(image)
            self.image_label.config(image=img)
            self.image_label.image = img
        except Exception as e:
            print("Error displaying image:", e)
            traceback.print_exc()

    def process_image(self):
        try:
            if not self.input_image:
                print("No image selected")
                return
            self.load_model()
            original_size = self.input_image.size
            width, height = original_size
            new_width = (width + 255) // 256 * 256
            new_height = (height + 255) // 256 * 256
            resized_image = self.input_image.resize((new_width, new_height))
            tensor = transform(resized_image).unsqueeze(0)
            with torch.no_grad():
                processed_tensor = model(tensor)
            processed_image = transforms.ToPILImage()(processed_tensor.squeeze(0))
            self.processed_image = processed_image
            self.display_image(self.processed_image)
        except Exception as e:
            print("Error processing image:", e)
            traceback.print_exc()
    
    def split_image(self, image):
        try:
            image = image.convert("RGB")
            patches = []
            positions = []
            for i in range(0, image.width, 256):
                for j in range(0, image.height, 256):
                    patch = image.crop((i, j, i + 256, j + 256))
                    if patch.size[0] < 256 or patch.size[1] < 256:
                        patch = self.pad_image(patch)
                    patches.append(patch)
                    positions.append((i, j))
            return patches, positions
        except Exception as e:
            print("Error splitting image:", e)
            traceback.print_exc()
    
    def pad_image(self, image):
        try:
            new_image = Image.new("RGB", (256, 256), (0, 0, 0))
            new_image.paste(image, (0, 0))
            return new_image
        except Exception as e:
            print("Error padding image:", e)
            traceback.print_exc()
    
    def process_patch(self, patch):
        try:
            tensor = transform(patch).unsqueeze(0)
            with torch.no_grad():
                processed_tensor = model(tensor)
            processed_patch = transforms.ToPILImage()(processed_tensor.squeeze(0))
            return processed_patch
        except Exception as e:
            print("Error processing patch:", e)
            traceback.print_exc()
    
    def merge_patches(self, patches, positions, original_size):
        try:
            merged_image = Image.new("RGB", original_size)
            for patch, (i, j) in zip(patches, positions):
                merged_image.paste(patch, (i, j))
            return merged_image
        except Exception as e:
            print("Error merging patches:", e)
            traceback.print_exc()
    
    def on_resize(self, event):
        try:
            if self.input_image and not self.processed_image:
                self.display_image(self.input_image)
            elif self.processed_image:
                self.display_image(self.processed_image)
        except Exception as e:
            print("Error on resize:", e)
            traceback.print_exc()

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg;*.jpeg")])
            if file_path:
                self.processed_image.save(file_path)
                print("Image saved to:", file_path)
        else:
            print("No image to save")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
