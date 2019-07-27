import torch
import torchvision
import numpy as np

from train import load_data, imshow
import torchvision.models as models

from lenet import LeNet
from vgg16 import VGG16
from resnet import ResNet
from densenet import DenseNet

def evalidation(model_name, testloader, classes, input_channel=3, self_define=True):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # load model parameter
    assert model_name in ["LeNet", "VGG16", "ResNet", "DenseNet"]
    param_path = "./model/%s_%s_parameter.pt" % (model_name, "define" if self_define else "official")
    print("load model parameter from %s" % param_path)
    if self_define:
        if model_name == "LeNet":
            net = LeNet(input_channel)
        elif model_name == "VGG16":
            net = VGG16(input_channel)
        elif model_name == "ResNet":
            net = ResNet(input_channel)
        elif model_name == "DenseNet":
            net = DenseNet(input_channel)
    else:
        if model_name == "LeNet":
            net = LeNet(input_channel)
        elif model_name == "VGG16":
            net = models.vgg16_bn(pretrained=False, num_classes=10)
        elif model_name == "ResNet":
            net = models.resnet50(pretrained=False, num_classes=10)
        elif model_name == "DenseNet":
            net = models.DenseNet(num_classes=10)


    net.load_state_dict(torch.load(param_path))
    net.eval()

    # predict
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

    # to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # evaluate
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(batch_size):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    print("\nEvery class precious: \n ",
          ' '.join("%5s : %2d %%\n" % (classes[i], 100 * class_correct[i]/class_total[i]) for i in range(len(classes))))
    print("\n%d images in all, Total precious: %2d %%"
          % (np.sum(class_total), 100 * np.sum(class_correct) / np.sum(class_total)))


if __name__ == "__main__":
    batch_size = 4
    # load data
    _, _, testloader, classes = load_data(batch_size)

    evalidation("ResNet", testloader, classes, self_define=True)




