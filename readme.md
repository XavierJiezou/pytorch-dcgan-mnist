![在这里插入图片描述](https://img-blog.csdnimg.cn/20210125211852298.png)
# 1. 学习目标
本教程教你如何使用`dcgan`训练`mnist`数据集，生成手写数字。
# 2. 环境配置
## 2.1. Python
请参考[官网](https://www.python.org/downloads/)安装。
## 2.2. Pytorch
请参考[官网](https://pytorch.org/get-started/locally/)安装。
## 2.3. Jupyter notebook
```bash
pip install jupyter
```
## 2.4. Matplotlib
```bash
pip install matplotlib
```
# 3. 具体实现
## 3.1. 导入模块
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
## 3.2. 设置随机种子
```python
# 设置随机种子，以便复现实验结果。
torch.manual_seed(0)
```
## 3.3. 超参数配置
- **dataroot**：存放数据集文件夹所在的路径
- **workers** ：数据加载器加载数据的线程数
- **batch_size**：训练的批次大小。
- **image_size**：训练图像的维度。默认是64x64。如果需要其它尺寸，必须更改$D$和$G$的结构，点击[这里](https://github.com/pytorch/examples/issues/70)查看详情
- **nc**：输入图像的通道数。对于彩色图像是3
- **nz**：潜在空间的长度
- **ngf**：与通过生成器进行的特征映射的深度有关
- **ndf**：设置通过鉴别器传播的特征映射的深度
- **num_epochs**：训练的总轮数。训练的轮数越多，可能会导致更好的结果，但也会花费更长的时间
- **lr**：学习率。DCGAN论文中用的是0.0002
- **beta1**：Adam优化器的参数beta1。论文中，值为0.5
- **ngpus**：可用的GPU数量。如果为0，代码将在CPU模式下运行；如果大于0，它将在该数量的GPU下运行
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
## 3.4. 数据集
使用`mnist`数据集，其中训练集**6**万张，测试集**1**万张，我们这里不是分类任务，而是使用`gan`的生成任务，所以就不分训练和测试了，全部图像都可以利用。
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
输出：
```python
Total Size of Dataset: 70000
```
## 3.5. 数据加载器
```python
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)
```
## 3.6. 选择训练设备
检测`cuda`是否可用，可用就用`cuda`加速，否则使用`cpu`训练。
```python
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
```
## 3.7. 训练数据可视化
```python
inputs = next(iter(dataloader))[0]
plt.figure(figsize=(10,10), dpi=100)
plt.title("Training Images")
plt.axis('off')
inputs = utils.make_grid(inputs[:100], nrow=10)
plt.imshow(inputs.permute(1, 2, 0))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210125205118234.png)
## 3.8. 权重初始化
在`dcgan`论文中，作者指出所有模型权重应当从均值为**0**，标准差为**0.02**的正态分布中随机初始化。==但这里不建议使用，亲测使用后效果很差==。
```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```
## 3.9. 生成器
生成器的结构：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210125123946836.png)
构建生成器类：
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
生成器实例化：
```python
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
# netG.apply(weights_init)
```
## 3.10. 判别器
构建判别器类：
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
判别器实例化：
```python
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
# netD.apply(weights_init)
```
## 3.11. 优化器和损失函数
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
## 3.12. 开始训练
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
输出：
```python
Starting Training Loop...
Epoch: [1/5] Step: [700/700] Loss-D: 8.8328 Loss-G: 5.1051 D(x): 0.9995 D(G(z)): [0.9989/0.0200] Time: 1m6s
Epoch: [2/5] Step: [700/700] Loss-D: 2.5174 Loss-G: 0.7627 D(x): 0.1362 D(G(z)): [0.0006/0.5085] Time: 1m8s
Epoch: [3/5] Step: [700/700] Loss-D: 0.0355 Loss-G: 4.4222 D(x): 0.9767 D(G(z)): [0.0113/0.0163] Time: 1m8s
Epoch: [4/5] Step: [700/700] Loss-D: 0.9482 Loss-G: 1.9022 D(x): 0.6590 D(G(z)): [0.3345/0.1798] Time: 1m8s
Epoch: [5/5] Step: [700/700] Loss-D: 0.0939 Loss-G: 3.1018 D(x): 0.9168 D(G(z)): [0.0025/0.0698] Time: 1m8s
```
## 3.13. 训练过程中的损失变化
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
## 3.14. 训练过程中的D(x)和D(G(z))变化
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
## 3.15. 可视化G的训练过程
```python
fig = plt.figure(figsize=(10, 10), dpi=100)
fig = plt.figure()
plt.axis("off")
ims = [[plt.imshow(item.permute(1, 2, 0), animated=True)] for item in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021012600255757.gif#pic_center)
# 4. 真图 vs 假图
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
<center>（左边是数据集中的真图，右边是生成器生成的假图）</center>

# 5. 温馨提示
本教程使用的是**1**张`GTX 1080 Ti`的显卡，训练一个**epoch**大概`1m8s`左右。虽然实验室有**8**张卡，但没必要都用，亲测多卡训练速度反而更慢，当然我这里说的是数据并行`DataParallel`。分布式`distributed`训练的话应该会快很多，但对于初学者来说不太建议使用，因为配置很麻烦。
# 6. 引用参考
> [https://blog.csdn.net/qq_42951560/article/details/110308336](https://blog.csdn.net/qq_42951560/article/details/110308336)