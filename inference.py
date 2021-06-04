import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.utils.inference_utils import run_model
from torchvision import transforms
from model import shufflenet, efficientnet

CLASSES = ['Battery', 'Clothing', 'Glass', 'Metal', 'Paper', 'Paperpack', 'Plastic', 'Plasticbag', 'Styrofoam']

class CustomImageFolder(ImageFolder):
    """ImageFolder with filename."""

    def __getitem__(self, index):
        img_gt = super(CustomImageFolder, self).__getitem__(index)
        fdir = self.imgs[index][0]
        fname = fdir.rsplit(os.path.sep, 1)[-1]
        return (img_gt + (fname,))

@torch.no_grad()
def inference(model, dataloader):
    result = {}
    model = model.to(device)
    model.eval()
    submission_csv = {}
    for img, _, fname in tqdm(dataloader):
        img = img.to(device)
        pred, enc_data = run_model(model, img)
        pred = torch.argmax(pred)
        submission_csv[fname[0]] = CLASSES[int(pred.detach())]

    result["macs"] = enc_data
    result["submission"] = submission_csv
    j = json.dumps(result, indent=4)
    save_path = './save/submission.csv'
    with open(save_path, 'w') as outfile:
        json.dump(result, outfile)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare datalaoder
    data_dir = '/opt/ml/input/data/test'

    img_size = 64

    data_transforms = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor()
        ])

    dataset = CustomImageFolder(root=data_dir, transform=data_transforms)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=8
    )

    # prepare model
    model = shufflenet.shufflenet_v2_x0_5()
    model.fc = nn.Linear(in_features=1024, out_features=9, bias=True)

    model.load_state_dict(torch.load('/opt/ml/code/save/model.pt', map_location=torch.device('cpu')))

    # inference
    inference(model, dataloader)
