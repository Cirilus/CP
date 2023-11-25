import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm import create_model
from torch import nn
from torchmetrics import Accuracy
import lightning as L
import numpy as np

list_classes = ['Бетон', 'Грунт', 'Дерево', 'Кирпич']


# Define custom dataset with albumentations support
class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, target


class ImageClassifier(L.LightningModule):
    def __init__(self, num_classes, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model('resnetv2_50.a1h_in1k', pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy('multiclass', num_classes=num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        transform_train = A.Compose([
            A.RandomResizedCrop(600, 600),
            A.HorizontalFlip(),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        train_dataset = AlbumentationsDataset('train/', transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        return train_loader

    def val_dataloader(self):
        transform_val = A.Compose([
            A.Resize(600, 600),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        val_dataset = AlbumentationsDataset('val/', transform=transform_val)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
        return val_loader


# Define the transformation
def get_transform():
    transform = A.Compose([
        A.Resize(600, 600),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform


# Function to predict a single image
def predict_single_image(image, model, transform):
    # Load and preprocess the image
    # image = Image.open(image_path).convert('RGB')
    image = transform(image=np.array(image))['image']
    image = image.unsqueeze(0)  # Add batch dimension

    # Ensure model is in evaluation mode
    model.eval()

    # Make the prediction
    with torch.no_grad():
        output = model(image)

    # Convert output probabilities to predicted class
    _, preds = torch.max(output, 1)
    return preds.item()


def classify_image(image) -> str:
    # image = Image.open(image_path).convert('RGB')

    checkpoint_path = 'ml/classification_model.ckpt'
    model = ImageClassifier.load_from_checkpoint(checkpoint_path)
    # Get transformation
    transform = get_transform()
    # Predict the class of a single image
    predicted_class_index = predict_single_image(image, model, transform)
    # Convert index to class label if you have a mapping
    return list_classes[predicted_class_index]


if __name__ == "__main__":
    print(classify_image('grunt.jpg'))
