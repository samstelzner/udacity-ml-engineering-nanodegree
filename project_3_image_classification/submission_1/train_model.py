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

def test(model, test_loader, loss_criterion, hook, args):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=loss_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss * 100.0 / len(test_loader)
    total_acc = running_corrects * 100.0 / len(test_loader)
    
    print("Loss: {:.2f}%, Accuracy: {:.2f}%\n".format(total_loss, total_acc))
    
    logger.info(
        "Test set: Loss: {:.2f}, Accuracy: {:.2f}%\n".format(
            total_loss, total_acc
        )
    )
    
# Include any debugging/profiling hooks.
    hook.set_mode(smd.modes.EVAL)

def train(model, train_loader, validation_loader, loss_criterion, optimizer, hook, args, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    epochs=args.epochs
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
        # Stop training when the validation accuracy stops increasing.
            running_loss = 0.0
            running_corrects = 0
            running_samples=0
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
            
    return model
    
    hook.set_mode(smd.modes.TRAIN)
    
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
    define the data loaders centrally for use in various functions?
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
    model = train(model, train_loader, validation_loader, loss_criterion, optimizer, hook, args, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, hook, args)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    
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
        '--epochs', type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
    )
    
    parser.add_argument(
        '--model-dir', type=str, default=os.environ["SM_MODEL_DIR"]
    )
    
    args=parser.parse_args()
    
    main(args)