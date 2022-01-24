from network.net import VGG, make_layers
from network.utils import Flatten, CatGAP, accuracy, imshow_transform, SaveFeatures

import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np



def vgg19():
    model = VGG(make_layers())
    return model


if __name__ == '__main__':
    model = vgg19()
    labelnumber = [0,1,2,3,4,5,6]
    model.classifier = nn.Sequential(
        CatGAP()
    )
    state_dict = torch.load('model_pth/best_acc.pth')
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
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
    # change batch size to 1 to grab one image at a time
    test_loader = torch.utils.data.DataLoader(test_valid, batch_size=1,
                                          shuffle=True,
                                          num_workers=0)
    step_cls_wtbox = {}
    for i in range(7):
        step_cls_wtbox[str(i)] = []

    cls_wtbox = {}
    for i in range(7):
        cls_wtbox[str(i)] = []
    with torch.no_grad():
        fulltime = 0
        for images, labels in test_loader:
            tmp = 0
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            labels_index = labels.detach().cpu().numpy().tolist()
            # outputs4 = model(images)
            outlist, scale_w = model(images)
            scale_w = scale_w.detach().squeeze().cpu().numpy()
            outputs4 = outlist[4]
            # step attention weights_save
            # for i in range(len(labels_index)):
            step_cls_wtbox[str(labels)].append(scale_w.tolist())

            # compute average
            for i in range(7):
                att_w = np.array(step_cls_wtbox[str(i)])
                att_w = att_w.sum(0) / len(step_cls_wtbox[str(i)])
                cls_wtbox[str(i)].append(att_w)
        print(cls_wtbox)


