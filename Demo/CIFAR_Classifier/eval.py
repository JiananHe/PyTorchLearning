import torch
import torchvision
import numpy as np

from train import load_data, imshow
from lenet import LeNet
from vgg16 import VGG16


def evalidation(model_name, testloader, classes, input_channel=3):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # load model parameter
    assert model_name in ["LeNet", "VGG16"]
    param_path = "./model/" + model_name + "_parameter.pt"
    if model_name == "LeNet":
        net = LeNet(input_channel)
    elif model_name == "VGG16":
        net = VGG16(input_channel)
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

    evalidation("VGG16", testloader, classes)




