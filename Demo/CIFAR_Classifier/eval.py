import torch
import torchvision
import numpy as np
from train import load_data, imshow, Net


if __name__ == "__main__":
    batch_size = 4
    # load data
    _, testloader, classes = load_data(batch_size)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # load gpu model to cpu
    net = torch.load("./model/model.pt")
    net.eval()
    net.to('cpu')

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # predict
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

    # evaluate
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
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




