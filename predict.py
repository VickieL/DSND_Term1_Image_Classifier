import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image
import random, os
import numpy as np
import json
from pathlib import Path

import helper
from workspace_utils import active_session
import functions
import argparse


if __name__ == "__main__":

    #with open('cat_to_name.json', 'r') as f:
    #    cat_to_name = json.load(f)
    
    current_dir = Path.cwd()
    
    parser = argparse.ArgumentParser()
    parser.description='arguments for training file'
    parser.add_argument("--save_dir", help="directory that save the checkpoint files",type=str, default=str(current_dir / 'model_checkpoint'))
    parser.add_argument("--chpt_fn", help="name for the checkpoint file",type=str, default='checkpoint.pth')
    parser.add_argument("--pre_dir", help="folder directory for prediction",type=str, default=str(current_dir / 'flowers' / 'test'))
    parser.add_argument("--pre_folder", help="which folder to predict",type=str, default='1')
    parser.add_argument("--topk", help="return top k possible classes",type=int, default=5)
    parser.add_argument("--category_names", help="category_names from cat_to_name.json",type=str, default='cat_to_name.json')
    args = parser.parse_args()
    print('arguments are',args)
    
    checkpoint_path = str(Path(args.save_dir) / args.chpt_fn)
    pre_folder = str(Path(args.pre_dir) / args.pre_folder)
    pre_img = random.choice(os.listdir(pre_folder))
    pre_img_path = pre_folder + '/' + pre_img
    print(pre_img_path)
    
    topk = args.topk
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # rebuild the model from checkpoint
    model_2, optimizer_2 = functions.load_checkpoint(checkpoint_path)
    print('load finished!')
    #print(model_2)

    # open an image for prediction
    im = Image.open(pre_img_path)
    #plt.imshow(im)
    # process the image to tensor
    im_ts = functions.process_image(im)
    #helper.imshow(im_ts)
    print("finish image process")

    # predict image
    probs, classes, index = functions.predict(im_ts, model_2, topk)
    print(f"top {topk:.0f} probabilities are: {list(map(lambda x:round(x, 4), probs)) }")
    print(f"top {topk:.0f} classes are: {classes}")
    print(f"top {topk:.0f} class index are: {index}")

    # get flower name
    classes_name = functions.idx_to_name(index, cat_to_name)
    print(f"top {topk:.0f} class names are: {classes_name}")
    