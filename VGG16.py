#!/usr/bin/env python
# coding: utf-8

# Note: in writing this notebook I referenced the following article on fine-tuning pretrained nets for clasification:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# set new parameter specifications for fine-tuning vgg to conduct binary
# classification
cwd = os.getcwd()
data_dir = cwd+"/hotdog-nothotdog"

model_name = "vgg16" # can be vgg11, vgg16 or vgg19

num_classes = 2

batch_size = 8

num_epochs = 10

feature_extract = True


# Specify a helper function for model training and validation
def train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, is_inception=False):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #print("val preds: " + str(preds.size()))
                #print("val labels: " + str(labels.data.size()))
                running_corrects += torch.sum(preds == labels.data)
            

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if(phase == "val"):
                train_corrects = 0
                for inputs, labels in dataloaders["train"]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print("train preds: " + str(preds.size()))
                    #print("train labels: " + str(labels.data.size()))
                    train_corrects += torch.sum(preds == labels.data)
                epoch_train_acc = train_corrects.double()/len(dataloaders["train"].dataset)
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                train_acc_history.append(epoch_train_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

# create helper function to specify that training should only happen for new layers
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Specify a helper function to initialize a VGG model 
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    
    if(model_name == "vgg11"):
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif(model_name == "vgg16"):
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif(model_name == "vgg19"):
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    else:
        print("Invalid model name")
        exit()
        
    return model_ft, input_size


# Initialize the model we plan to use
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
print(model_ft)


# specify data loading procedures
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)




# Now, actually run the model

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist, train_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


# Initialize the non-pretrained version of the model used for this run
scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()
_,scratch_hist, scratch_train_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []
thist = []
s_thist = []

ohist = [h.cpu().numpy() for h in hist]
thist = [h.cpu().numpy() for h in train_hist]
shist = [h.cpu().numpy() for h in scratch_hist]
s_thist = [h.cpu().numpy() for h in scratch_train_hist]

plt.title("Pretrained vs Scratch Test Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Test Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.savefig("Pretrained vs Scratch_" + model_name + "_" + str(num_epochs) + ".png")
plt.show()
 
plt.clf()

plt.title("Test vs Train  Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Test Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Test")
plt.plot(range(1,num_epochs+1),thist,label="Train")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.savefig("Test vs Train_" + model_name + "_" + str(num_epochs) + ".png")
plt.show()

plt.clf()

plt.title("Scratch Test vs Train  Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Test Accuracy")
plt.plot(range(1,num_epochs+1),shist,label="Test")
plt.plot(range(1,num_epochs+1),s_thist,label="Train")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.savefig("Scratch Test vs Train_" + model_name + "_" + str(num_epochs) + ".png")
plt.show()

# print data
out_file = "data_" + model_name + "_" + str(num_epochs) + ".txt"
with open(out_file, 'w') as f:
    f.write('Original Validation Accuracy: ' + str(hist)+"\n")
    f.write('Original Train Accuracy: ' + str(train_hist)+"\n")
    f.write('Scratch Validation Accuracy: ' + str(scratch_hist)+"\n")
    f.write('Scratch Train Accuracy: ' + str(scratch_train_hist)+"\n")


    



# save model
#save_path = cwd+"trained_model_"+str(num_epochs)
#torch.save(model_ft.state_dict(), save_path)


# test loading the model
#model = models.vgg11_bn(pretrained=True)
#model.load_state_dict(torch.load(save_path))
#model.eval()





