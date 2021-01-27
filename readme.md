![在这里插入图片描述](https://img-blog.csdnimg.cn/20210125211852298.png)
# 1. Target
Based on the mnist dataset, we will train a `dcgan` to generate new handwritten digit.
# 2. Environment
## 2.1. Python
download: [https://www.python.org/downloads/](https://www.python.org/downloads/)
## 2.2. Pytorch
download: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
## 2.3. Jupyter notebook
```bash
pip install jupyter
```
## 2.4. Matplotlib
```bash
pip install matplotlib
```
# 3. Implementation
## 3.1. Import modules
```python
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
```
## 3.2. Randomseed
```python
# Set random seed for reproducibility
torch.manual_seed(0)
```
## 3.3. Hyperparameter
- **dataroot**：the path to the root of the dataset folder. We will talk more about the dataset in the next section
- **workers** ：the number of worker threads for loading the data with the DataLoader
- **batch_size**：the batch size used in training.
- **image_size**：the spatial size of the images used for training. This implementation defaults to 64x64. If another size is desired, the structures of D and G must be changed. See [here](https://github.com/pytorch/examples/issues/70) for more details
- **nc**：number of color channels in the input images. For color images this is 3
- **nz**：length of latent vector
- **ngf**：relates to the depth of feature maps carried through the generator
- **ndf**：sets the depth of feature maps propagated through the discriminator
- **num_epochs**：number of training epochs to run. Training for longer will probably lead to better results but will also take much longer
- **lr**：learning rate for training. As described in the DCGAN paper, this number should be 0.0002
- **beta1**：beta1 hyperparameter for Adam optimizers. As described in paper, this number should be 0.5
- **ngpus**：number of GPUs available. If this is 0, code will run in CPU mode. If this number is greater than 0 it will run on that number of GPUs
```python
# Root directory for dataset
dataroot = "data/mnist"

# Number of workers for dataloader
workers = 12

# Batch size during training
batch_size = 100

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
```
## 3.4. Dataset
```python
mnist_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(
    root=dataroot,
    train=True,
    transform=mnist_transform
    download=True
)
test_data = datasets.MNIST(
    root=dataroot,
    train=False,
    transform=mnist_transform
)
dataset = train_data+test_data
print(f'Total Size of Dataset: {len(dataset)}')
```
out：
```python
Total Size of Dataset: 70000
```
## 3.5. Dataloader
```python
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)
```
## 3.6. Device
```python
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
```
## 3.7. Training Images Visualization
```python
inputs = next(iter(dataloader))[0]
plt.figure(figsize=(10,10), dpi=100)
plt.title("Training Images")
plt.axis('off')
inputs = utils.make_grid(inputs[:100], nrow=10)
plt.imshow(inputs.permute(1, 2, 0))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210125205118234.png)
## 3.8. Weight Initialization
From the DCGAN paper, the authors specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02. But it is not recommended to the mnist dataset.
```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```
## 3.9. Generator
Structure of Generator：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210125123946836.png)
Code：
```python
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```
Instantiation of Generator：
```python
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
# netG.apply(weights_init)
```
## 3.10. Discriminator
Code：
```python
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # state size. (1) x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```
Instantiation of Discriminator：
```python
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
# netD.apply(weights_init)
```
## 3.11. Optimizers and Loss Functions
```python
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(100, nz, 1, 1, device=device)
# print(f'Size of Latent Vector: {fixed_noise.size()}')

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```
## 3.12. Training
```python
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    beg_time = time.time()
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0) # 64*8
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1) # output.size()=[128]

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        end_time = time.time()
        run_time = round(end_time-beg_time)
        print(
            f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
            f'Step: [{i+1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
            f'Loss-D: {errD.item():.4f}',
            f'Loss-G: {errG.item():.4f}',
            f'D(x): {D_x:.4f}',
            f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
            f'Time: {int(run_time/60)}m{run_time%60}s',
            end='\r'
        )

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Save D(X) and D(G(z)) for plotting later
        D_x_list.append(D_x)
        D_z_list.append(D_G_z2)
        

        # Check how the generator is doing by saving G's output on fixed_noise
        iters += 1
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(utils.make_grid(fake, nrow=10))
    print()
```
out：
```python
Starting Training Loop...
Epoch: [1/5] Step: [700/700] Loss-D: 8.8328 Loss-G: 5.1051 D(x): 0.9995 D(G(z)): [0.9989/0.0200] Time: 1m6s
Epoch: [2/5] Step: [700/700] Loss-D: 2.5174 Loss-G: 0.7627 D(x): 0.1362 D(G(z)): [0.0006/0.5085] Time: 1m8s
Epoch: [3/5] Step: [700/700] Loss-D: 0.0355 Loss-G: 4.4222 D(x): 0.9767 D(G(z)): [0.0113/0.0163] Time: 1m8s
Epoch: [4/5] Step: [700/700] Loss-D: 0.9482 Loss-G: 1.9022 D(x): 0.6590 D(G(z)): [0.3345/0.1798] Time: 1m8s
Epoch: [5/5] Step: [700/700] Loss-D: 0.0939 Loss-G: 3.1018 D(x): 0.9168 D(G(z)): [0.0025/0.0698] Time: 1m8s
```
## 3.13. Loss versus training iteration
```python
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses[::100], label="G")
plt.plot(D_losses[::100], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210125220249298.png)
## 3.14. D(x) and D(G(z)) versus training iteration
```python
plt.figure(figsize=(10, 5))
plt.title("D(x) and D(G(z)) During Training")
plt.plot(D_x_list[::100], label="D(x)")
plt.plot(D_z_list[::100], label="D(G(z))")
plt.xlabel("iterations")
plt.ylabel("Probability")
plt.legend()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210125232842936.png)
## 3.15. Visualization of G’s progression
```python
fig = plt.figure(figsize=(10, 10), dpi=100)
fig = plt.figure()
plt.axis("off")
ims = [[plt.imshow(item.permute(1, 2, 0), animated=True)] for item in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021012600255757.gif#pic_center)
# 4. Real Images vs. Fake Images
```python
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(20,10), dpi=300)
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(utils.make_grid(real_batch[0][:100], nrow=10).permute(1, 2, 0))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.savefig('comparation.jpg', )
plt.imshow(transforms.Normalize((0.1307,), (0.3081,))(img_list[-1]).permute(1, 2, 0))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210126004054782.png)

# 5. Cite
> [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)