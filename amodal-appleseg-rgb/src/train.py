import torch
import torch.nn as nn
from PIL import Image
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import segmentation
from torchvision.models.segmentation import DeepLabV3
from dataset import AppleSegmentationDataset

transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class AppleSegmentationDeepLabV3:
    def __init__(self, dataset: Dataset, dataloader: DataLoader, model: DeepLabV3, lr: float = 0.001):
        if not isinstance(model, DeepLabV3):
            raise TypeError(f'Invalid model instance ({type(model)}) please use a DeepLabV3 instance')

        self.dataset = dataset
        self.dataloader = dataloader
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = BCEWithLogitsLoss()
        self.lr = lr

        in_features = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=(1, 1))
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
    
    def to_device(self):
        self.model = self.model.to(self.device)
        return self
    
    def train(self, epochs: int = 10):
        for epoch in range(epochs):
            print(f'Starting epoch')
            self.model.train()
            running_loss = 0.0

            for images, masks in self.dataloader:
                images, masks = images.to(self.device), masks.to(self.device)
                masks = masks.to(torch.float32)

                self.optimizer.zero_grad()

                outputs = self.model(images)['out']

                loss = self.criterion(outputs, masks)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss/len(self.dataloader)}')

train_dataset = AppleSegmentationDataset('../02-annotated_data_fuji/images/train', 'amodal', transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)

pretrained_weights = segmentation.DeepLabV3_ResNet101_Weights.DEFAULT

model = segmentation.deeplabv3_resnet101(weights=pretrained_weights)

ASDLV3 = AppleSegmentationDeepLabV3(train_dataset, train_loader, model)

ASDLV3.to_device().train()
torch.save(ASDLV3.model.state_dict(), 'deeplabv3_apples.pth')