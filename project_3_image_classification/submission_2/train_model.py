# Import Pytorch dependencies.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import smdebug.pytorch as smd

# https://knowledge.udacity.com/questions/32899
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_data_loaders(data_dir, batch_size, shuffle=True):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    )
    
# https://knowledge.udacity.com/questions/751453 - os.environ variables (SM_CHANNEL_TRAIN etc.)
# https://knowledge.udacity.com/questions/924037 - creating data loaders
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html - thorough example
    _set = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(_set, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def test(model, test_loader, loss_criterion, hook, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    # 2nd attempt
    print("START TESTING")
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_criterion(output, target)
            test_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, validation_loader, loss_criterion, optimizer, args, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # 2nd attempt:
    print("START TRAINING")
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader, 1):
# Zero out gradients from previous step
        optimizer.zero_grad()
        data=data.to(device)
        target=target.to(device)
# Forward pass
        output=model(data)
# Calculate loss
        loss=loss_criterion(output, target)
# Backward pass
        loss.backward()
# Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 500 == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
                
# perhaps worth adding a validation step to the above?
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
# densenet, a good performer in the article below for this kind of dataset, that is not resnet, which we've been practising with
# https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a
# densenet121 is the simplest version, let's start there

# proving to be buggy, rather testing with resnet first (most familiar from course: https://learn.udacity.com/nanodegrees/nd189/parts/cd0387/lessons/611008cc-58c1-4716-a98e-f4c1932668c6/concepts/b003fec9-c353-4646-b3b4-480b4c634deb)
    model = models.resnet18(pretrained=True)
    
# freeze the convolutional layers so we can finetune
    for param in model.parameters():
        param.requires_grad = False
        
# add a fully connected layer on top
# there are 133 dog breeds (classes)
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    
    return model

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
#   Training our model will go faster if we use the GPU.  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model=model.to(device)
    
    '''
    define the data loaders centrally for use in various functions
    '''
    train_loader = create_data_loaders(data_dir=args.train, batch_size=args.batch_size)
    validation_loader = create_data_loaders(data_dir=args.val, batch_size=args.batch_size, shuffle=False)
    test_loader = create_data_loaders(data_dir=args.test, batch_size=args.batch_size)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    '''
    define hook in main? https://learn.udacity.com/nanodegrees/nd189/parts/cd0387/lessons/29c31245-eeb5-412b-a09b-ec6e06df10fc/concepts/6140c483-a41b-49da-9af2-41e5a284fce2
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)    
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
#     ref: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-debugger/pytorch_model_debugging/scripts/pytorch_mnist.py
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, validation_loader, loss_criterion, optimizer, args, device, hook)
        test(model, test_loader, loss_criterion, hook, device)
    
    '''
    TODO: Save the trained model
    '''
#     done in 'train' function
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser.add_argument(
        "--batch-size", type=int, default=1, metavar="N", help="input batch size for training (default: 1)"
    )
    
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        '--train', type=str, default=os.environ['SM_CHANNEL_TRAIN']
    )
    
    parser.add_argument(
        '--val', type=str, default=os.environ['SM_CHANNEL_VAL']
    )
    
    parser.add_argument(
        '--test', type=str, default=os.environ['SM_CHANNEL_TEST']
    )
    
    parser.add_argument(
        '--epochs', type=int, default=3, metavar="N", help="number of epochs to train (default: 1)"
    )
    
    parser.add_argument(
        '--model-dir', type=str, default=os.environ["SM_MODEL_DIR"]
    )
    
    args=parser.parse_args()
    
    main(args)