import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob
# direct model from google
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        n = 32
        self.e_conv1 = nn.Conv2d(3, n, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(n, n, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(n, n, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(n, n, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(n * 2, n, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(n * 2, n, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(n * 2, 24, 3, 1, 1)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (x ** 2 - x)
        x = x + r2 * (x ** 2 - x)
        x = x + r3 * (x ** 2 - x)
        enhance_image_1 = x + r4 * (x ** 2 - x)
        x = enhance_image_1 + r5 * (enhance_image_1 ** 2 - enhance_image_1)
        x = x + r6 * (x ** 2 - x)
        x = x + r7 * (x ** 2 - x)
        enhance_image = x + r8 * (x ** 2 - x)
        return enhance_image
class LOLDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_paths = sorted(glob.glob(os.path.join(low_dir, "*.png")))
        self.high_paths = sorted(glob.glob(os.path.join(high_dir, "*.png")))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.low_paths)
    def __getitem__(self, idx):
        low = Image.open(self.low_paths[idx]).convert("RGB")
        high = Image.open(self.high_paths[idx]).convert("RGB")
        return self.transform(low), self.transform(high)
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = enhance_net_nopool().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = LOLDataset(
        low_dir=r"C:\Users\vedhr\datasets\our485\low",
        high_dir=r"C:\Users\vedhr\datasets\our485\high")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for low_img, high_img in tqdm(loader, desc=f"Epoch {epoch+1}"):
            low_img, high_img = low_img.to(device), high_img.to(device)
            enhanced = model(low_img)
            loss = criterion(enhanced, high_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "zero_dce_trained.pth")

if __name__ == "__main__":
    train()

