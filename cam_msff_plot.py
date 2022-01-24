import os
import math
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import utils as vutils
from torchvision import datasets, transforms

from network.net import VGG, make_layers
from network.utils import CatGAP, imshow_transform, SaveFeatures
from skimage.transform import resize


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def vgg19(load_state_dict=False):
    model = VGG(make_layers())
    if load_state_dict:
        state_dict = torch.load('./vgg19-model.pth')
        model.load_state_dict(state_dict)
    return model


def plot(bee_ant, bee_cam, path, cnt, cls):
    plt.figure()
    plt.subplots(figsize=(4, 4))
    plt.imshow(bee_ant)
    plt.imshow(bee_cam, alpha=.4, cmap='jet')
    plt.savefig(path+'/CAM_img_%s' % cnt+'_class_%s' % cls+'.jpg')
    plt.close()
    # plt.show()


def getAnsCam(img_path, filename, tmpsf, outputs, count, trans_img, cls):
    if os.path.exists(img_path + '/%s' % filename):
        pass
    else:
        os.mkdir(img_path + '/%s' % filename)
    path = img_path + '/%s' % filename
    tmpsf.remove()
    arr = tmpsf.features.cpu().detach().numpy()
    arr1 = arr[0]
    ans_list = []
    ans_list.append(np.dot(np.rollaxis(arr1, 0, 3), [1, 0, 0, 0, 0, 0, 0]))
    ans_list.append(np.dot(np.rollaxis(arr1, 0, 3), [0, 1, 0, 0, 0, 0, 0]))
    ans_list.append(np.dot(np.rollaxis(arr1, 0, 3), [0, 0, 1, 0, 0, 0, 0]))
    ans_list.append(np.dot(np.rollaxis(arr1, 0, 3), [0, 0, 0, 1, 0, 0, 0]))
    ans_list.append(np.dot(np.rollaxis(arr1, 0, 3), [0, 0, 0, 0, 1, 0, 0]))
    ans_list.append(np.dot(np.rollaxis(arr1, 0, 3), [0, 0, 0, 0, 0, 1, 0]))
    ans_list.append(np.dot(np.rollaxis(arr1, 0, 3), [0, 0, 0, 0, 0, 0, 1]))
    for i in range(7):
        imshow(ans_list[i], cmap='jet')
        plt.savefig(path + '/ans_%s' % i + filename)
        plt.close()

    cls_cam_list = []
    for i in range(7):
        cls_cam_list.append(resize(ans_list[i], (224, 224)))
        imshow(cls_cam_list[i], cmap='jet')
        plt.savefig(path + '/ans_%s_upsample' % i + filename)
        plt.close()

    ans_all = np.zeros((224, 224))
    for i in range(7):
        weight = outputs[0][i].detach().cpu().numpy()
        weight = math.exp(weight)
        ans_all += weight * cls_cam_list[i]
    plot(trans_img, ans_all, path, count, cls)
    return ans_all


def normalize(ans):
    max = ans.max()
    min = ans.min()
    ans = (ans-min)/(max-min)
    return ans


def getFusionCam(model):
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_test = datasets.ImageFolder(root='./dataset/test/',
                                         transform=valid_transforms)

    # change batch size to 1 to grab one image at a time
    test_loader = torch.utils.data.DataLoader(test_test, batch_size=1,
                                          shuffle=True,
                                          num_workers=0)
    count = 0

    for i, (images, labels, sample_fname) in enumerate(test_loader, 0):
        sf = model.classifier[0]
        sf0 = SaveFeatures(sf.Conv3)
        sf1 = SaveFeatures(sf.Conv8)
        sf2 = SaveFeatures(sf.Conv17)
        sf3 = SaveFeatures(sf.Conv26)
        sf4 = SaveFeatures(sf.Conv35)
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        # params = list(model.parameters())

        outlist = model(images)

        im = images
        lab = labels.cpu().detach().numpy()

        os.mkdir('E:\\CAM_plot\\' + sample_fname[0].split('\\')[1] + '-cls%s' % lab)
        img_path = 'E:\\CAM_plot\\' + sample_fname[0].split('\\')[1] + '-cls%s' % lab
        # im = Variable(im.cuda())


        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224).cuda()
        t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224).cuda()
        input_tensor = images * t_std + t_mean
        input_tensor = input_tensor.clone().detach()
        # 到cpu
        input_tensor = input_tensor.to(torch.device('cpu'))
        # input_tensor = unnormalize(input_tensor)
        if os.path.exists(img_path+'/'+sample_fname[0].split('\\')[1]):
            pass
        else:
            vutils.save_image(input_tensor, img_path+'/'+sample_fname[0].split('\\')[1])

        outputs0 = outlist[0]
        outputs1 = outlist[1]
        outputs2 = outlist[2]
        outputs3 = outlist[3]
        outputs4 = outlist[4]

        cls = lab

        trans_img = imshow_transform(im)

        # last layer
        ans_all0 = getAnsCam(img_path, 'Conv3', sf0, outputs0, count, trans_img, cls)
        ans_all1 = getAnsCam(img_path, 'Conv8', sf1, outputs1, count, trans_img, cls)
        ans_all2 = getAnsCam(img_path, 'Conv17', sf2, outputs2, count, trans_img, cls)
        ans_all3 = getAnsCam(img_path, 'Conv26', sf3, outputs3, count, trans_img, cls)
        ans_all4 = getAnsCam(img_path, 'Conv35', sf4, outputs4, count, trans_img, cls)
        relu_ans_all0 = np.maximum(ans_all0, 0)
        relu_ans_all1 = np.maximum(ans_all1, 0)
        relu_ans_all2 = np.maximum(ans_all2, 0)
        relu_ans_all3 = np.maximum(ans_all3, 0)
        relu_ans_all4 = np.maximum(ans_all4, 0)

        ans_final = normalize(relu_ans_all0)+normalize(relu_ans_all1)+normalize(relu_ans_all2)+normalize(relu_ans_all3) + normalize(relu_ans_all4)

        path = os.path.join(img_path, "CAM_MSFF")
        os.makedirs(path, exist_ok=True)

        plot(trans_img, ans_final, path, count, cls)
        count += 1


if __name__ == '__main__':
    model = vgg19()
    # modify the last two convolutions
    model.features[-5] = nn.Conv2d(512, 512, 3, padding=1)
    model.features[-3] = nn.Conv2d(512, 512, 3, padding=1)

    # modeify classfier
    model.classifier = nn.Sequential(
        CatGAP()
    )

    state_dict = torch.load('model_pth/best_acc.pth')
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    getFusionCam(model)

















