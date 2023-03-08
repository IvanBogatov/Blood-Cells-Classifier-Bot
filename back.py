from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms as T

from config import CLASSES, WEIGHTS_PATH

# version problem solution
import collections
collections.Iterable = collections.abc.Iterable


# Model
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.layer1 = nn. Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2,2))
    
    self.layer2 = nn. Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2,2))
    
    self.layer3 = nn. Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2,2))
    
    self.layer4 = nn. Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2,2))
    
    self.fc=nn.Sequential(
        nn.Linear(256*17*17,512),
        nn.ReLU(),
        nn.Linear(512,4)
        )

    self.drop_out = nn.Dropout()   

  def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)        
        x=self.layer3(x) 
        x=self.layer4(x) 
        x = self.drop_out(x)      

        x=x.view(-1,256*17*17)
        x=self.fc(x)
        
        return x

# Image transformation
def trnsfrms(img): 
    return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5,0.5,0.5])
                ])(img)

def img_handler(img_io):
    # Open image
    img = Image.open(img_io)
    img = trnsfrms(img)

    # Init model and update weights
    model = CNN()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()

    # Get best 2 guesses
    probs = torch.softmax(model(img)[0], dim=0).detach().tolist()
    idxes = [probs.index(i) for i in sorted(probs, reverse=True)[:2]]
    return CLASSES[idxes[0]], probs[idxes[0]], CLASSES[idxes[1]], probs[idxes[1]]