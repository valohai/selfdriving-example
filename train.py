import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import cv2
import numpy as np
import csv
from model import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--max_epochs',
    type=int,
    default=20,
    help='Number of epochs to run trainer',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Number of steps to run trainer',
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.001,
    help='Initial learning rate',
)
parser.add_argument(
    '--validation_set_size',
    type=float,
    default=0.2,
    help='Percentage of steps to use for validation vs. training',
)
flags = parser.parse_args()


# Step1: Read from the log file
samples = []
with open('/valohai/inputs/driving_dataset/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

# Step2: Divide the data into training set and validation set
train_len = int((1.0 - flags.validation_set_size)*len(samples))
valid_len = len(samples) - train_len
train_samples, validation_samples = data.random_split(samples, lengths=[train_len, valid_len])

# Step3a: Define the augmentation, transformation processes, parameters and dataset for dataloader
def augment(imgName, angle):
  name = '/valohai/inputs/driving_dataset/data/IMG/' + imgName.split('/')[-1]
  current_image = cv2.imread(name)
  current_image = current_image[65:-25, :, :]
  if np.random.rand() < 0.5:
    current_image = cv2.flip(current_image, 1)
    angle = angle * -1.0
  return current_image, angle


class Dataset(data.Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        steering_angle = float(batch_samples[3])
        center_img, steering_angle_center = augment(batch_samples[0], steering_angle)
        left_img, steering_angle_left = augment(batch_samples[1], steering_angle + 0.4)
        right_img, steering_angle_right = augment(batch_samples[2], steering_angle - 0.4)
        center_img = self.transform(center_img)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)
        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)

    def __len__(self):
        return len(self.samples)

# Step3b: Creating generator using the dataloader to parallasize the process
transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)])

params = {'batch_size': flags.batch_size,
          'shuffle': True,
          'num_workers': 4}

training_set = Dataset(train_samples, transformations)
training_generator = DataLoader(training_set, **params)

validation_set = Dataset(validation_samples, transformations)
validation_generator = DataLoader(validation_set, **params)


# Step5: Define optimizer
model = NetworkLight()
optimizer = optim.Adam(model.parameters(), lr=flags.learning_rate)
criterion = nn.MSELoss()

# Step6: Check the device and define function to move tensors to that device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is: ', device)


def toDevice(datas, device):
    imgs, angles = datas
    return imgs.float().to(device), angles.float().to(device)


max_epochs = flags.max_epochs
for epoch in range(max_epochs):
    model.to(device)

    # Training
    train_loss = 0
    valid_loss = 0
    real_train_loss = 0
    real_valid_loss = 0

    model.train()
    for local_batch, (centers, lefts, rights) in enumerate(training_generator):
        # Transfer to GPU
        centers, lefts, rights = toDevice(centers, device), toDevice(lefts, device), toDevice(rights, device)

        # Model computations
        optimizer.zero_grad()
        datas = [centers, lefts, rights]
        for data in datas:
            imgs, angles = data
            outputs = model(imgs)
            loss = criterion(outputs, angles.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.data

        real_train_loss = train_loss / (local_batch + 1)
            
    # Validation
    model.eval()
    with torch.set_grad_enabled(False):
        for local_batch, (centers, lefts, rights) in enumerate(validation_generator):
            # Transfer to GPU
            centers, lefts, rights = toDevice(centers, device), toDevice(lefts, device), toDevice(rights, device)

            # Model computations
            optimizer.zero_grad()
            datas = [centers, lefts, rights]
            for data in datas:
                imgs, angles = data
                outputs = model(imgs)
                loss = criterion(outputs, angles.unsqueeze(1))
                valid_loss += loss.data

            real_valid_loss = valid_loss / (local_batch + 1)

    print('{"loss": %f, "valid_loss": %f, "epoch": %s}' % (real_train_loss, real_valid_loss, epoch))

# Step8: Define state and save the model wrt to state
state = {'model': model.module if device == 'cuda' else model}

torch.save(state, '/valohai/outputs/model.h5')
