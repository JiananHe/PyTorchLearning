import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import os

from lenet import LeNet
from vgg16 import VGG16


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        size  (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        interpolation(int, optional): Desired interpolation. Default is "PIL.Image.BILINEAR"
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, (int, tuple))
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled. img.format, img.size, img.mode

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size[:2]
        if isinstance(self.size, int):
            if h > w:
                new_w = self.size
                new_h = new_w * h / w
            else:
                new_h = self.size
                new_w = new_h * w / h
        else:
            new_h, new_w = self.size, self.size

        return img.resize((int(new_h), int(new_w)), self.interpolation)


def load_data(batch_size):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    # Composes several transforms together.
    # Normalize: input[channel] = (input[channel] - mean[channel]) / std[channel]
    transform_train = transforms.Compose([Rescale(32),
                                          # transforms.Resize(32),  # img should be PIL Image.
                                          transforms.RandomHorizontalFlip(),  # img should be PIL Image.
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_val)
    # trainset = torchvision.datasets.MNIST(root='./data', train=True,
    #                                       download=True, transform=transform)
    # testset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                      download=True, transform=transform)

    # randomly splitting validation and test dataset indices
    set_len = len(testset)
    indices = list(range(set_len))
    val_ratio = 0.2
    val_len = int(val_ratio * set_len)
    np.random.seed(1000)
    val_indices = np.random.choice(indices, size=val_len, replace=False)  # replace=false, 抽样后不放回，若为true，可能有重复
    test_indices = list(set(indices) - set(val_indices))

    # num_workers: 设置多进程提取数据
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2, sampler=SubsetRandomSampler(val_indices))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2, sampler=SubsetRandomSampler(test_indices))

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, validloader, testloader, classes


def imshow(img):
    print(img.shape)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # transpose dims of array
    plt.show()


def valid(model, validloader, criterion, device):
    # switch model to evaluation mode
    model.eval()

    valid_loss = 0.0
    n = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            n += 1

    return valid_loss / n


def training(model_name, trainloader, validloader, input_channel=3, epochs=1, resume=True):
    # create net
    assert model_name in ["LeNet", "VGG16"]
    if model_name == "LeNet":
        net = LeNet(input_channel)
    elif model_name == "VGG16":
        net = VGG16(input_channel)

    # resume training
    if resume:
        param_path = "./model/" + model_name + "_parameter.pt"
        if os.path.exists(param_path):
            net.load_state_dict(torch.load(param_path))
            net.train()
            print("Resume training" + model_name)
        else:
            print("Train %s from scratch" % model_name)
    else:
        print("Train %s from scratch" % model_name)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('train on %s' % device)
    net.to(device)

    running_loss = 0.0
    train_losses = []
    valid_losses = []
    mini_batches = 125 * 5
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            # get one batch
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # switch model to training mode, clear gradient accumulators
            net.train()
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % mini_batches == mini_batches - 1:  # print and valid every <mini_batches> mini-batches
                # validate model in validation dataset
                valid_loss = valid(net, validloader, criterion, device)
                print('[%d, %5d] train loss: %.3f,  validset loss: %.3f' % (
                    epoch + 1, i + 1, running_loss / mini_batches, valid_loss))
                train_losses.append(running_loss / mini_batches)
                valid_losses.append(valid_loss)
                running_loss = 0.0

        # # save checkpoint
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': net.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss
        # }, "./checkpoints/epoch_" + str(epoch) + ".tar")

    print('Finished Training, %d images in all' % (len(train_losses) * batch_size * mini_batches / epochs))

    # draw loss curve
    assert len(train_losses) == len(valid_losses)
    loss_x = range(0, len(train_losses))
    plt.plot(loss_x, train_losses, label="train loss")
    plt.plot(loss_x, valid_losses, label="valid loss")
    plt.title("Loss for every %d mini-batch" % mini_batches)
    plt.xlabel("%d mini-batches" % mini_batches)
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_name + "_loss.png")
    plt.show()

    # save parameters
    torch.save(net.state_dict(), "./model/" + model_name + "_parameter.pt")
    # save the whole model
    # torch.save(net, "./model/model.pt")


if __name__ == "__main__":
    batch_size = 16
    # download and set data loader
    trainloader, validloader, testloader, classes = load_data(batch_size)

    # show some images
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()  # get four(batch_size) samples
    # images
    imshow(torchvision.utils.make_grid(images))
    # labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    training("LeNet", trainloader, validloader)
