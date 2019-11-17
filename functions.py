import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image
import numpy as np
import json

import helper
from workspace_utils import active_session

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data function
# input : train, valid, test data path. like 'flowerz/train'
# output: trainloader, testloader, validloader, train_data(for class_to_idx)
def load_data(train_dir, valid_dir, test_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    return trainloader, testloader, validloader, train_data


# check whether data is loaded succesfully
# input : trainloader / testloader / validloader, show picture #
# output: show first x pictures
def dataloader_show(dataloader, pic_number):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    fig, axes = plt.subplots(figsize=(10,4), ncols=pic_number)
    for ii in range(pic_number):
        ax = axes[ii]
        helper.imshow(images[ii], ax=ax, normalize=True)


# test model using all testing data
# input : model, testloader
# output: print the Valid loss and Valid accuracy
# return: N/A
def validation_all_test_data(model, testloader):
    model.eval()  
    test_loss = 0
    accuracy_test = 0
    criterion = nn.NLLLoss()

    for images_test, labels_test in testloader:
        images_test, labels_test = images_test.to(device), labels_test.to(device)
        logps_test = model(images_test)
        loss_test = criterion(logps_test, labels_test)  
        test_loss += loss_test.item()  
                    
        # calculate our accuracy
        ps = torch.exp(logps_test) 
        top_p, top_class = ps.topk(1, dim=1)               
        equality = top_class == labels_test.view(*top_class.shape) 
        accuracy_test += torch.mean(equality.type(torch.FloatTensor)).item() 

    # average accuracy
    print(f"Valid loss: {test_loss/len(testloader):.3f}.. "   
          f"Valid accuracy: {accuracy_test/len(testloader):.3f}")   


# loads a checkpoint and rebuild model
# input : filepath (i.e. 'checkpoint_vgg16.pth')
# return: model, optimizer
def load_checkpoint(filepath):
    # 加载checkpoint文件
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    # initiate model
    #model = models.vgg16()  
    model = getattr(models, checkpoint['arch'])(pretrained=True)
   
    # define model using checkpoint info
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.loss = checkpoint['loss']
    
    # load model
    model.load_state_dict(checkpoint['state_dict'])
    #model.to(device)
    print("model loaded successfully")
       
    # initiate optimizer
    optimizer = optim.Adam(model.classifier.parameters())
    print(optimizer.state_dict()['param_groups'])
    print(checkpoint['optimizer_state_dict']['param_groups'])
    
    # load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    print("optimizer loaded successfully")
    
    return model, optimizer


# Process a PIL image for use in a PyTorch model
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
# input : image <class 'PIL.JpegImagePlugin.JpegImageFile'> (601, 500, 3)
# return: image_tensor <class 'torch.Tensor'> torch.Size([3, 224, 224]) (color channel to be the first dimension)
def process_image(image):
    print(type(image))
    transform_process = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),  
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])
    image_tensor = transform_process(image)
    print(type(image_tensor))
    
    return image_tensor


# can use helper.imshow() instead.
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# Predict the class (or classes) of an image using a trained deep learning model.
# input : img_ts, model, topk(top possible classes)
# return: top_p, top_class(folder index), top_index(class_to_idx)
def predict(img_ts, model, topk):
    
    #model.eval()
    img_ts = img_ts.view(1, 3, 224, 224)
    img_ts = img_ts.to(device)
    model = model.to(device)
    
    ps = torch.exp(model(img_ts))
    print(ps.shape)
    top_p, top_class = ps.topk(topk, dim=1) #这里得到的class是文件夹的顺序，不是文件夹名称
                                            #（数据集的label也是表示文件夹的位置，而不是文件夹的名称。所以如果有测试集的类别顺序和训练集不一样，就会有问题）
    top_p = top_p.tolist()[0]
    top_class = top_class.tolist()[0]
    
    # 需要将class转化成index（后面需要用index来map类别的名称）
    # 获取class_to_idx的内容（class和index的转化关系）
    idx = []
    for i in range(len(model.class_to_idx.items())):
        idx.append(list(model.class_to_idx.items())[i][0])
    
    # class to index
    top_index = []
    for i in range(topk):
        top_index.append(idx[top_class[i]])
    
    return top_p, top_class, top_index


# Get flower name from index
# input : index
# return: class name from cat_to_name.json file
def idx_to_name(index, cat_to_name):
    classes_name = []
    for i in range(len(index)):
        classes_name.append(cat_to_name[index[i]])
        
    return classes_name


# *****NOT USED**********probabilities for the top 5 classes as a bar graph, along with the input image.
# input : index
# output: show the picture and bar graph
# return: N/A
def predict_display(title, image, classes_name_sort, probs_sort):
    plt.figure(figsize=(6, 12))

    fig,ax=plt.subplots(2,1)
    # define fist picture
    ax[0].set_title(title)
    ax[0].axis('off')
    ax[0].imshow(image)

    # define bar graph
    y = np.arange(len(classes_name_sort))
    x = probs_sort
    ax[1].barh(y,x,align='center') 
    plt.yticks(y,classes_name_sort)

    plt.tight_layout()
    plt.show()
