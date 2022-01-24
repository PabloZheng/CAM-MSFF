from sklearn import metrics
import time
from network.net import VGG, make_layers
from network.utils import Flatten, accuracy, imshow_transform, SaveFeatures
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from metric import make_confusion_matrix


def vgg19():
    model = VGG(make_layers())
    return model


if __name__ == '__main__':
    model = vgg19()
    labelnumber = [0, 1, 2, 3, 4, 5, 6]
    model.classifier = nn.Sequential(
                                    nn.Conv2d(512, 7, 1, padding=0),
                                    nn.AdaptiveAvgPool2d(1), Flatten(),
                                    nn.Softmax()
                                    )
    state_dict = torch.load('model_pth/best_acc.pth')
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    correct = 0
    total = 0
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    allpre = []
    alllabel = []
    filenames = []

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_valid = datasets.ImageFolder(root='./dataset/test/',
                                         transform=valid_transforms)
    test_loader = torch.utils.data.DataLoader(test_valid, batch_size=1,
                                          shuffle=True,
                                          num_workers=0)

    with torch.no_grad():
        fulltime = 0
        for images, labels in test_loader:
            tmp = 0
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs4 = model(images)
            t1 = time.time()
            predicted = torch.argmax(outputs4.data)
            t2 = time.time()
            tmp = t2 - t1
            fulltime+=tmp
            p = predicted.cpu().detach().numpy()
            allpre.append(p)
            l = labels.detach().cpu().numpy()
            lab = l[0]
            # l = l.tolist()
            alllabel.append(l)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # c = (predicted == labels).squeeze()

            if predicted == labels:
                class_correct[lab] += 1
                class_total[lab] += 1
    print('full time and average time :',fulltime, (fulltime/total))
    print('Accuracy of the network on the images : %6f %%' % (
            100 * correct / total))
    for i in range(len(labelnumber)):
        print('Accuracy of %5s : %6f %%' % (
            labelnumber[i], 100 * class_correct[i] / (class_total[i]+1e-9)))

    testresult={'filename': filenames, 'pre': allpre, 'true': alllabel}
    print(metrics.f1_score(alllabel, allpre, labelnumber, average=None))
    print(metrics.classification_report(alllabel, allpre, labels=labelnumber, digits=7))
    make_confusion_matrix(alllabel, allpre, labelnumber)



