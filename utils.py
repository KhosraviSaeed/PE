"""
Pose Detection utils.py
Saeed Khosravi

"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

import torchvision.models as models
import torchvision.transforms as transforms

import os
import time

im_size = 300 # to change images size to im_size * im_size

class lsb_dataset(Dataset):
    """
    In Classification and Localization problems that inputs of a sample
    are more than one or our labels are formatted in a specific way it is
    needed to make a class to return samples as they are.
    For instance in Face Recognition problem we need three images in a sample
    an Anchor, a negative image and a positive image therefore our getitem returns
    a sample which are three images
    """
    def __init__(self, imgs, jnts):
        self.imgs = imgs
        self.jnts = jnts
        self.x_locs = jnts[0,:,:]
        self.y_locs = jnts[1,:,:]
        self.are_vis = jnts[2,:,:]
        self.tfms = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __getitem__(self, index):
        image = Image.open(self.imgs[index]).convert('RGB')
        w, h = image.size
        image = self.tfms(image)
        x_loc = self.x_locs[:,index].reshape(len(self.x_locs[:,index]),1)
        x_loc = x_loc * (im_size/w)
        y_loc = self.y_locs[:,index].reshape(len(self.y_locs[:,index]),1)
        y_loc = y_loc * (im_size/h)
        is_vis = self.are_vis[:,index].reshape(len(self.are_vis[:,index]),1)
        jnt_loc = np.zeros((28,1))
        jnt_loc[:14,0], jnt_loc[14:,0]  = x_loc.reshape((14,)),  y_loc.reshape((14,))
        jnt_loc[:14,0], jnt_loc[14:,0] = jnt_loc[:14,0] / im_size, jnt_loc[14:,0] / im_size
        return image, torch.Tensor(jnt_loc)
        
    def __len__(self):
        return len(self.imgs)

    
def show_im_jnts(image, jnt_loc, title):
    """
    plot image with the joints and their names
    
    """
    img = image.numpy().transpose((1, 2, 0))  # (H, W, C)
    # denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    fig, ax = plt.subplots(figsize=(14,10))
    plt.title(title)
    plt.imshow(img)
    jnt_show = jnt_loc
    jnt_show[:14,0], jnt_show[14:,0] = jnt_loc[:14,0] * im_size, jnt_loc[14:,0] * im_size
    ax.scatter(jnt_show[:14,0],jnt_show[14:,0],color='r')
    txt = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle'
    , 'Right wrist', 'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow', 'Left wrist'
    , 'Neck', 'Head top']
    for i in range(14):
        ax.annotate(txt[i], (jnt_loc[i,0], jnt_loc[14+i,0]))
        
def imshow(imgs, jnt_locs, title=None):
    """Imshow for Tensor.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    imgs = imgs.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    imgs = std * imgs + mean
    imgs = np.clip(imgs, 0, 1)
    jnt_show = jnt_locs
    for i in range(8):
        jnt_show[i,:14,0], jnt_show[i,14:,0] = jnt_locs[i,:14,0] * im_size, jnt_locs[i,14:,0] * im_size
        plt.scatter( jnt_show[i,:14,0] + (i%4)*299 + (i%4) * 4, jnt_show[i,14:,0] + int(i/4)*299, marker='o', c='red')
    plt.imshow(imgs)
    plt.axis('off')
    if title is not None:
        plt.title(title)
        
def imshow_one(img, jnt_locs, title=None):
    """Imshow for Tensor plotting one image.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    plt.imshow(img)
    ax.scatter(coord[:14,0],coord[14:,0], c='r' )
    plt.axis('off')
    if title is not None:
        plt.title(title)
    txt = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle'
    , 'Right wrist', 'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow', 'Left wrist'
    , 'Neck', 'Head top']
    for i in range(14):
        ax.annotate(txt[i], (jnt_locs[i,0], jnt_locs[14+i,0] ))


class my_FCN(nn.Module):
    """
    Making a Fully Convolutional CNN, using a pre-trained resnet CNN
    by removing the last fully connected layer and adding a linear
    layer
    
    """
    def __init__(self):
        super(my_FCN, self).__init__()
        res101 = models.resnet101(pretrained=True)
        num_features = res101.fc.in_features
        res101.fc = nn.Linear(num_features, 28)
        self.model = res101
    def forward(self, x):
        x= x.float()
        x = self.model(x)
        coords = F.sigmoid(x)
        return coords
    

class my_LLoss(nn.Module):
    """

    Calculating the loss between actual joints location and coordinates
    that gets training by the model. Loss function is Mean Square Error
    but instead of deviding to N to calculate the Mean I divided it to
    2 to take a bigger error which should be optimized.
    
    """
    def __init__(self):
        super(my_LLoss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)
    def forward(self, coords, jnt_locs):
        coords = coords.reshape((8,28,1))
        loss = 0
        for i in range(8):
            for j in range(28):
                loss += self.mse_loss(coords[i,j,0],jnt_locs[i,j,0])/2.0

        return loss
    
    
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler):
    """
    This function train the model one epoch by calculating the forward using
    the model and then pytorch calculates the hard part(backward)
    and then we update the parameters by optimizer.step()
    
    """
    if scheduler is not None:
        scheduler.step()
    
    model.train(True)
    
    steps = len(dataloader.dataset) // dataloader.batch_size
    
    for i, (imgs, jnt_locs) in enumerate(dataloader):
        inputs, jnt_locs = Variable(imgs), Variable(jnt_locs)
        optimizer.zero_grad()
        
        # forward
        coords = model(inputs)
        coords = coords.float()
        #score, loc, is_vis, jnt_loc
        loc_loss = criterion(coords, jnt_locs)        
        loss = 10.0 * loc_loss
        print(f'Batch {i} of {steps} loss : {loss}')
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


def train_model(model, train_dl, valid_dl, criterion, optimizer,
                scheduler=None, num_epochs=10):

    if not os.path.exists('models'):
        os.mkdir('models')
    
    since = time.time()
       
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # train
        model = train_one_epoch(model, train_dl, criterion, optimizer, scheduler)
        # I do not validate the model to train faster

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
        
