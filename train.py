import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from model import shufflenet, efficientnet
from sklearn.metrics import f1_score
from src.utils import loss, macs
from tqdm import tqdm
import os
import copy
import time

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            labels_total = []
            preds_total = []

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                labels_total.extend(labels.cpu().numpy().tolist())
                preds_total.extend(preds.cpu().numpy().tolist())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(labels_total,preds_total, average='macro')

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc, epoch_f1))

            if phase == 'val':
                scheduler.step()

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "./save/model.pt")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))

    model.load_state_dict(best_model_wts)
    return model

class Efficientnet(nn.Module):
    def __init__(self):
        super(Efficientnet, self).__init__()
        self.net = efficientnet.efficientnet_b0(pretrained=True)
        self.linear = nn.Linear(1000,9)

    def forward(self,x):
        x = self.net(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":

    # https://pytorch.org/vision/stable/transforms.html

    img_size = 32

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([img_size,img_size]),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4991, 0.4795, 0.4673), (0.2048, 0.2043, 0.2123))
        ]),
        'val': transforms.Compose([
            transforms.Resize([img_size,img_size]),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4966, 0.4769, 0.4646), (0.2057, 0.2053, 0.2126))
        ]),
    }

    data_dir = '/opt/ml/input/data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=256,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # https://pytorch.org/vision/stable/models.html

    model = shufflenet.shufflenet_v2_x0_5(pretrained=True)
    model.conv5 = nn.Sequential(
        nn.Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )
    model.fc = nn.Linear(in_features=128, out_features=9, bias=True)
    # model.load_state_dict(torch.load('/opt/ml/code/save/model.pt'))
    model = model.to(device)

    criterion = loss.LabelSmoothingLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5)

    print('MACs :', macs.calc_macs(model, (3, img_size, img_size)))

    # train_model(model, criterion, optimizer, scheduler, num_epochs=50)
