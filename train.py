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
import time
from pathlib import Path

import helper
from workspace_utils import active_session
import functions
import argparse

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    current_dir = Path.cwd()
    
    parser = argparse.ArgumentParser()
    parser.description='arguments for training file'
    parser.add_argument("--data_dir", help="directory of images",type=str, default=str(current_dir / 'flowers'))
    parser.add_argument("--save_dir", help="directory that save the checkpoint files",type=str, default=str(current_dir / 'model_checkpoint'))
    parser.add_argument("--chpt_fn", help="name for the checkpoint file",type=str, default='checkpoint.pth')
    parser.add_argument("--arch", help="model architecture",type=str, default='vgg16')
    parser.add_argument("--drop_p", help="dropout percentage",type=float, default=0.5)
    parser.add_argument("--learn_r", help="learning rate",type=float, default=0.01)
    parser.add_argument("--hid_u", help="hidden layer units",type=int, default=1024)
    parser.add_argument("--epochs", help="epochs for training",type=int, default=5)
    args = parser.parse_args()
    print('arguments are',args)
    
    arch = args.arch
    dropout_p = args.drop_p
    learn_rate = args.learn_r
    hidden_units = args.hid_u
    class_num = 102

    epochs = args.epochs
    steps = 0
    train_loss = 0
    print_every = 30

    data_dir = Path(args.data_dir)
    train_dir = str(data_dir / 'train')
    valid_dir = str(data_dir / 'valid')
    test_dir = str(data_dir / 'test')
    
    save_dir = str(Path(args.save_dir) / args.chpt_fn)
    
    # load data
    trainloader, testloader, validloader, train_data = functions.load_data(train_dir, valid_dir, test_dir)

    # load the model from torchvision
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Define our new classifier
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                            nn.Dropout(p=dropout_p),
                            nn.ReLU(),
                            nn.Linear(hidden_units, class_num),
                            nn.LogSoftmax(dim=1))   
    model.classifier = classifier

    # Define loss / optimizer
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    optimizer = optim.SGD(model.classifier.parameters(), lr = learn_rate)

    model.to(device)

    # train classifier
    start = time.time()
    print('start training')
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            
            # 把张量转移到GPU上
            images, labels = images.to(device), labels.to(device)
            
            # 编写training循环
            # 首先，清零梯度，这步很重要，不要忘了
            optimizer.zero_grad()
            
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()  # 优化步骤
            
            train_loss += loss.item()  # 递增train_loss，这样当我们用越来越多当数据训练后，可以跟踪训练损失。
            
            # print("steps={steps}".format(steps=steps))
            
            # 跟踪验证集的损失和准确率，以确定最佳超参数
            if steps % print_every == 0:
                model.eval()  # 将模型变成评估推理模型，这样会关闭丢弃
                valid_loss = 0
                accuracy = 0
                
                print("steps={steps}".format(steps=steps))
                
                for images, labels in validloader:
                    
                    # 把张量转移到GPU上
                    images, labels = images.to(device), labels.to(device)
            
                    logps = model(images)
                    loss = criterion(logps, labels)
                    
                    valid_loss += loss.item()  # 跟踪测试损失
                    
                    # calculate our accuracy
                    ps = torch.exp(logps) #模型返回的是logSoftmax，表示类别的对数概率，要得到实际的概率，要用.exp()
                    
                    top_ps, top_class = ps.topk(1, dim=1)
                    
                    equality = top_class == labels.view(*top_class.shape) #检查结果是否与lables匹配
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item() #根据equality张量更新accuracy
            
                print(f"Epoch {epoch+1}/{epochs}.. "  #看看where we are
                    f"Train loss: {train_loss/print_every:.3f}.. "   #对训练损失取平均值
                    f"Valid loss: {valid_loss/len(validloader):.3f}.. "   #len(validloader)表示我们从validloader获得了多少批数据
                    f"Valid accuracy: {accuracy/len(validloader):.3f}"   #测试集的平均准确率
                )
                
                train_loss = 0
                model.train()  #改回train模式

    traing_time = time.time() - start
    print("finish training!")
    print("Total time: {:.0f}m {:.0f}s".format(traing_time//60, traing_time % 60))

    # do validation on test data
    functions.validation_all_test_data(model, testloader)

    # save the model to checkpoint
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'loss': train_loss,
                  'arch': args.arch
            }
    torch.save(checkpoint, save_dir)
    print("save checkpoint file succesfully!")
    