

import os
import time
import h5py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from google.colab import drive


# 1. SETUP AND EXTRACTION

try:
    from google.colab import drive
    drive.mount('/content/drive')
    zip_path = '/content/drive/MyDrive/dataset.zip'
    base_extract_dir = '/content/dataset_local'
except ModuleNotFoundError:
    print("Running locally. Skipping Google Drive mount.")
    zip_path = './dataset.zip'
    base_extract_dir = './dataset_local'

check_folder = os.path.join(base_extract_dir, 'shanghaitech_with_people_density_map')

if os.path.exists(check_folder):
    print("Dataset already extracted! Skipping unzip process.")
else:
    print(f"Dataset not found at {check_folder}. Extracting now...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_extract_dir)
    print("Extraction complete!")

# 2. DATASET CLASS

class ShanghaiTechDataset(Dataset):
    def __init__(self, image_dir, gt_dir, target_size=(768, 1024)):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.target_size = target_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]

        # Load Image
        img_path = os.path.join(self.image_dir, file_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        img_tensor = self.transform(img)

        # Load Density Map
        h5_name = file_name.replace('.jpg', '.h5')
        h5_path = os.path.join(self.gt_dir, h5_name)
        with h5py.File(h5_path, 'r') as hf:
            density_map = np.asarray(hf['density'])
            density_map = cv2.resize(density_map, self.target_size)

        density_tensor = torch.from_numpy(density_map).unsqueeze(0).float() * 100.0
        return img_tensor, density_tensor


print("Setting up Unified Data Loaders...")
base_path = '/content/dataset_local/shanghaitech_with_people_density_map/ShanghaiTech'

# --- TRAINING DATA ---
train_A = ShanghaiTechDataset(f'{base_path}/part_A/train_data/images', f'{base_path}/part_A/train_data/ground-truth-h5')
train_B = ShanghaiTechDataset(f'{base_path}/part_B/train_data/images', f'{base_path}/part_B/train_data/ground-truth-h5')
combined_train = ConcatDataset([train_A, train_B])
train_loader = DataLoader(combined_train, batch_size=4, shuffle=True)

# --- TESTING DATA ---
test_A = ShanghaiTechDataset(f'{base_path}/part_A/test_data/images', f'{base_path}/part_A/test_data/ground-truth-h5')
test_B = ShanghaiTechDataset(f'{base_path}/part_B/test_data/images', f'{base_path}/part_B/test_data/ground-truth-h5')
combined_test = ConcatDataset([test_A, test_B])
test_loader = DataLoader(combined_test, batch_size=1, shuffle=False)

print(f"Total Training Images: {len(combined_train)} | Total Testing Images: {len(combined_test)}")

#  MODEL ARCHITECTURES

class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3), nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2), nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(nn.Conv2d(30, 1, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        x1, x2, x3 = self.branch1(x), self.branch2(x), self.branch3(x)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        return nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if load_weights else None)
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[0:23])
        for param in self.frontend.parameters():
            param.requires_grad = False

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return nn.functional.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)

#  TRAINING ENGINE - MCNN ONLY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware active: {device}\n")

criterion = nn.MSELoss()
num_epochs = 100

print("--- Starting 100-Epoch Training for MCNN ---")
mcnn_model = MCNN().to(device)
mcnn_optimizer = optim.Adam(mcnn_model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    epoch_start = time.time()
    mcnn_model.train()
    running_loss = 0.0

    for batch_idx, (images, densities) in enumerate(train_loader):
        images, densities = images.to(device), densities.to(device)

        mcnn_optimizer.zero_grad()
        predictions = mcnn_model(images)
        loss = criterion(predictions, densities)
        loss.backward()
        mcnn_optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"MCNN Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.6f} | Time: {time.time() - epoch_start:.2f}s")


torch.save(mcnn_model.state_dict(), '/content/drive/MyDrive/mcnn_final_weights.pth')
print("MCNN Training Complete & Saved to Drive!\n")


del mcnn_model
del mcnn_optimizer
torch.cuda.empty_cache()
print("GPU Memory Cleared. Ready for CSRNet (or a break if Colab limits you).")

#  TRAINING ENGINE - CSRNET ONLY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware active: {device}\n")

criterion = nn.MSELoss()
num_epochs = 50

print("--- Starting 50-Epoch Training for CSRNET ---")
csrnet_model = CSRNet().to(device)
csrnet_optimizer = optim.Adam(csrnet_model.backend.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    epoch_start = time.time()
    csrnet_model.train()
    running_loss = 0.0

    for batch_idx, (images, densities) in enumerate(train_loader):
        images, densities = images.to(device), densities.to(device)

        csrnet_optimizer.zero_grad()
        predictions = csrnet_model(images)
        loss = criterion(predictions, densities)
        loss.backward()
        csrnet_optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"CSRNet Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.6f} | Time: {time.time() - epoch_start:.2f}s")


torch.save(csrnet_model.state_dict(), '/content/drive/MyDrive/csrnet_final_weights.pth')
print("CSRNet Training Complete & Saved to Drive!")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader, ConcatDataset


print("Setting up Unified Test Data...")
base_path = '/content/dataset_local/shanghaitech_with_people_density_map/ShanghaiTech'

test_A = ShanghaiTechDataset(
    image_dir=os.path.join(base_path, 'part_A/test_data/images'),
    gt_dir=os.path.join(base_path, 'part_A/test_data/ground-truth-h5')
)
test_B = ShanghaiTechDataset(
    image_dir=os.path.join(base_path, 'part_B/test_data/images'),
    gt_dir=os.path.join(base_path, 'part_B/test_data/ground-truth-h5')
)

combined_test = ConcatDataset([test_A, test_B])
test_loader = DataLoader(combined_test, batch_size=1, shuffle=True)
print(f"Total Test Images Available: {len(combined_test)}")

#  LOAD TRAINED MODELS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")


mcnn_weights_path = '/content/drive/MyDrive/mcnn_final_weights.pth'
csrnet_weights_path = '/content/drive/MyDrive/csrnet_final_weights.pth'

# Load MCNN
mcnn_eval = MCNN().to(device)
mcnn_eval.load_state_dict(torch.load(mcnn_weights_path, map_location=device))
mcnn_eval.eval()

# Load CSRNet
csrnet_eval = CSRNet(load_weights=False).to(device)
csrnet_eval.load_state_dict(torch.load(csrnet_weights_path, map_location=device))
csrnet_eval.eval()

print("Models loaded successfully!")

#  EVALUATION METRICS

print("\nRunning models on Test Dataset...")

with torch.no_grad():
    for images, densities in test_loader:
        images = images.to(device)
        densities = densities.to(device)

        gt_count = torch.sum(densities).item()/100.0

        mcnn_pred = mcnn_eval(images)
        mcnn_count = torch.sum(mcnn_pred).item()/100.0

        csr_pred = csrnet_eval(images)
        csr_count = torch.sum(csr_pred).item()/100.0

        break # Breaking early for the visual test

 VISUALIZATION

print("\nGenerating Comparison Visuals...\n")

original_img = images[0].cpu().permute(1, 2, 0).numpy()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
original_img = std * original_img + mean
original_img = np.clip(original_img, 0, 1)

gt_map = densities[0].cpu().squeeze().numpy()
mcnn_map = mcnn_pred[0].cpu().squeeze().numpy()
csr_map = csr_pred[0].cpu().squeeze().numpy()

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(original_img)
axes[0].set_title("Original Image", fontsize=14)
axes[0].axis('off')

axes[1].imshow(gt_map, cmap='jet')
axes[1].set_title(f"Ground Truth\nActual Count: {gt_count:.1f}", fontsize=14)
axes[1].axis('off')

axes[2].imshow(mcnn_map, cmap='jet')
axes[2].set_title(f"MCNN (Baseline)\nPredicted Count: {mcnn_count:.1f}", fontsize=14)
axes[2].axis('off')

axes[3].imshow(csr_map, cmap='jet')
axes[3].set_title(f"CSRNet (Advanced)\nPredicted Count: {csr_count:.1f}", fontsize=14)
axes[3].axis('off')

plt.tight_layout()
plt.show()