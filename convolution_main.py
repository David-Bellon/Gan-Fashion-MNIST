import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import numpy as np

to_pil = ToPILImage()
device = torch.device("cuda")
batch_size = 120


train_set = DataLoader(
    torchvision.datasets.FashionMNIST("data", download=True, transform=ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 784)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 32),
            nn.LeakyReLU(),

            nn.Linear(32, 16),
            nn.LeakyReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        out = self.model(x)
        return out


generator = Generator().to(device)
discrimator = Discriminator().to(device)
criterion = nn.BCELoss()
g_optim = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optim = torch.optim.Adam(discrimator.parameters(), lr=0.0002)


def train_discriminator(real_images, fake_images):
    d_optim.zero_grad()

    real_label = torch.ones(batch_size, 1).to(device)
    fake_label = torch.zeros(batch_size, 1).to(device)

    real_out = discrimator(real_images)
    fake_out = discrimator(fake_images)

    real_loss = criterion(real_out, real_label)
    fake_loss = criterion(fake_out, fake_label)

    real_loss.backward()
    fake_loss.backward()

    d_optim.step()

    return real_loss + fake_loss


def train_generator(real_images):
    g_optim.zero_grad()

    real_label = torch.ones((batch_size, 1)).to(device)
    out = discrimator(real_images)

    loss = criterion(out, real_label)

    loss.backward()
    g_optim.step()

    return loss


def create_noise():
    return torch.randn((batch_size, 128)).to(device)


latent = torch.randn((64, 128)).to(device)
epochs = 100
images = []
for epoch in range(epochs):
    gen_loss = 0
    dis_loss = 0
    for i, data in tqdm(enumerate(train_set)):
        real_images = data[0]
        real_images = real_images.to(device)
        fake_images = generator(create_noise())
        gen_loss = gen_loss + train_discriminator(real_images, fake_images)
        fake_images = generator(create_noise())
        dis_loss = dis_loss + train_generator(fake_images)

    gen_image = generator(latent).detach().cpu()
    gen_image = make_grid(gen_image)
    images.append(gen_image)

    print(f"Epoch: {epoch} ----- Discriminator Loss: {dis_loss/i} ----- Generator Loss: {gen_loss/i}")

imgs = [np.array(to_pil(img)) for img in images]
imageio.mimsave('outs.gif', imgs)
