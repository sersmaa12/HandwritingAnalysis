
# import os
# import torch
# from torch import optim
# from torch.utils.data import DataLoader
# import argparse
# from datetime import datetime
# from AuthorsDataset import AuthorsDataset, Pad, Threshold, ShiftAndCrop, Downsample
# from BaselineSiamese import BaselineSiamese, BaselineNet

# # ----------------------------
# # Command line аргументууд
# # ----------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("data_path", type=str, help="Path to train.txt file")
# parser.add_argument("-c", "--cuda", action="store_true", help="Use GPU if available")
# parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of training epochs")
# parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
# args = parser.parse_args()

# # ----------------------------
# # Hyperparameters & Dataset transforms
# # ----------------------------
# MAXWIDTH = 2552     # Original image width
# MAXHEIGHT = 1457    # Original image height
# IMG_CROP = 700      # Crop width for CNN
# DOWNSAMPLE = 0.75   # Resize factor
# BATCH_SIZE = 10

# transform_pipeline = transforms.Compose([
#     Pad((MAXWIDTH, MAXHEIGHT)),
#     Threshold(177),
#     ShiftAndCrop(IMG_CROP, random=True),
#     Downsample(DOWNSAMPLE),
# ])

# # Dataset & DataLoader
# train_dataset = AuthorsDataset(
#     root_dir='Dataset',
#     path=args.data_path,
#     transform=transform_pipeline
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# # ----------------------------
# # Model, Loss, Optimizer
# # ----------------------------
# model = BaselineSiamese()  # or BaselineNet() if only one branch

# criterion = torch.nn.CrossEntropyLoss()
# # optimizer = optim.SGD(model.parameters(), lr=1, weight_decay=0.01) 
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)

# # Load checkpoint if exists
# if args.load_checkpoint:
#     checkpoint = torch.load(args.load_checkpoint)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])

# # CUDA setup
# device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
# model.to(device)

# # ----------------------------
# # Training Loop
# # ----------------------------
# log_file = open("loss_log.txt", "a")

# for epoch in range(args.epochs):
#     epoch_loss = []

#     for batch_idx, (X1, X2, Y) in enumerate(train_loader):
#         X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)

#         optimizer.zero_grad()
#         Y_hat = model(X1, X2)

#         loss = criterion(Y_hat, Y)
#         loss.backward()
#         optimizer.step()

#         epoch_loss.append(loss.item())
#         print(f"EPOCH: {epoch} | BATCH: {batch_idx} | LOSS: {loss.item():.6f}")

#     avg_loss = sum(epoch_loss) / len(epoch_loss)
#     print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")
#     log_file.write(f"{datetime.now()} Epoch {epoch} Loss {avg_loss:.6f}\n")

#     # Save checkpoint every 5 epochs
#     if epoch % 5 == 0:
#         os.makedirs("Model_Baseline_Checkpoints", exist_ok=True)
#         checkpoint_path = os.path.join("Model_Baseline_Checkpoints", f"epoch{epoch}.pt")
#         torch.save({
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict()
#         }, checkpoint_path)

# log_file.close()



from torch.utils.data import DataLoader
import os
import sys
#sys.path.append('../')
from AuthorsDataset import *
from torch import optim
from torchvision import transforms
import torchvision
from torch import norm
import numpy as np
from BaselineSiamese import *
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("-e", "--epochs", type=int, default=20)
parser.add_argument("--load_checkpoint", type=str, default=None)
args = parser.parse_args()

# Initialize model, loss and optimizer
model = BaselineSiamese()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 1, weight_decay=0.01)
#optimizer = optim.Adam(model.parameters())

if args.load_checkpoint:
    checkpoint_path = args.load_checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Initialize cuda
if args.cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model.cuda()

# Constants from Authors100 dataset
MAXWIDTH = 2260
MAXHEIGHT = 337

train_dataset = AuthorsDataset(
    root_dir='Dataset',
    path=args.data_path,
    transform=transforms.Compose([
        Pad((MAXWIDTH, MAXHEIGHT)),
        Threshold(177),
        ShiftAndCrop(700, random=True),
        Downsample(0.75),
    ]))

train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True
)

file_ = open("loss_50000.txt", "a")

for epoch in range(args.epochs):
    batch_loss = []
    for batch_idx,(X1,X2,Y) in enumerate(train_loader):

        # Move batch to GPU
        
        # Move data to GPU
        if args.cuda:
            X1,X2,Y = X1.to(device),X2.to(device),Y.to(device)

        # Compute forward pass
        Y_hat = model.forward(X1,X2)


        # Perform backprop and zero gradient
        optimizer.zero_grad()
        # Apply Forward pass on input pairs
        Y_hat = model.forward(X1,X2)
        
        # Calculate loss on training data
        loss = criterion(Y_hat, Y)
       
        # Back propagate the loss and apply zero grad
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss.append(loss.item())
        print("EPOCH: %d\t BATCH: %d\tTRAIN LOSS = %f"%(epoch,batch_idx,loss.item()))
    file_.write(str(loss.item())+"\n")        

    print (sum(batch_loss) / len(batch_loss))
    # Save model checkpoints
    if epoch%5 == 0:
        checkpoint_path = os.path.join('Model_Baseline_Checkpoints',"epoch" + str(epoch))
        checkpoint = {'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}

        torch.save(checkpoint, checkpoint_path)
