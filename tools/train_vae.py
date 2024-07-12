import torch
import torchvision.utils

from dataloader.mnist_dataloader import MnistDataset
from model.vanilla_vae import VAEmodel
from torch.utils.data.dataloader import DataLoader
from einops import rearrange
from torch.optim import Adam
from tqdm import tqdm
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_vae():
    # Create the dataloader and dataloader
    mnist = MnistDataset('train', 'data/train/images/')
    mnist_test = MnistDataset('test', 'data/test/images')
    mnist_loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=0)

    # Instantiate the model
    model = VAEmodel().to(device)

    # Specify the training parameters
    num_epochs = 10
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    recon_losses = []
    kl_losses = []
    losses = []

    # Train the model for 10 epochs
    for epoch_idx in range(num_epochs):
        for im, label in tqdm(mnist_loader):
            im = im.float().to(device)
            optimizer.zero_grad()

            #print(f'label image shape is {label.shape}')
            mean, log_var, out = model(im)
            cv2.imwrite('input.jpeg', 255*((im + 1)/2).detach().cpu().numpy()[0, 0])
            cv2.imwrite('output.jpeg', 255 * ((out + 1) / 2).detach().cpu().numpy()[0, 0])

            #print(f'im shape is {im.shape}')
            kl_loss = torch.mean(torch.sum(torch.exp(log_var) + mean**2 -1 -log_var, dim=-1))
            recon_loss = criterion(out, im)
            loss = recon_loss + 0.00001*kl_loss

            losses.append(loss.detach().cpu().numpy())
            kl_losses.append(kl_loss.detach().cpu().numpy())
            recon_losses.append(recon_loss.detach().cpu().numpy())

            # Backpropagation
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch_idx+1} | Recon Loss is {np.mean(recon_losses):.4f} | KL Loss is {np.mean(kl_losses):.4f}')
    print('Done Training...!')

    # Reconstruct for some random images
    idxs = torch.randint(0, len(mnist_test)-1, (100, ))
    ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()  # mdoel expects a batch of images

    _, _, generated_im = model(ims)

    # visualization
    ims = (ims + 1)/2
    generated_ims = 1 - (generated_im+1)/2

    out = torch.hstack([ims, generated_ims])
    output = rearrange(out, pattern='b c h w -> b () h (c w)')
    grid = torchvision.utils.make_grid(output, nrow=10)

    img = torchvision.transforms.ToPILImage()(grid)
    img.save('reconstructeed_img.jpeg')
    print('Done Reconstruction...!')
    pass

if __name__ == '__main__':
    train_vae()