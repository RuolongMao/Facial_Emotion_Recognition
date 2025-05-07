# Facial Emotion Recognition (FER) Project

This project builds neural networks that produces accurate predictions about human emotion images, and provides visualizations about facial features that the model focuses on while training.
The project includes multiple implementations of facial emotion recognition models:
- BaselineCNN (55% accuracy)
- ImprovedBaselineCNN (62% accuracy)
- ResNet50-based model
- Final optimized model with GradCAM visualization (77% accuracy)

## Runnable Scripts and Commands

### Setup and Installation
```bash
# Create a virtual environment (recommended)
python -m venv fer_env
source fer_env/bin/activate  # On Windows: fer_env\Scripts\activate

# Install required packages
pip install torch torchvision numpy matplotlib scikit-learn opencv-python grad-cam
```

### Running the Models

0. **Data Pipeline**
```python
# data loading
uploaded = files.upload()

import zipfile
import io

zip_name = next(iter(uploaded))
with zipfile.ZipFile(io.BytesIO(uploaded[zip_name]), 'r') as zip_ref:
    zip_ref.extractall('/content/')

# finalized data augmentation(included in training scripts)
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

2. **BaselineCNN Model (55% accuracy)**
```python
# In BaselineCNN - 55%.ipynb
# Key configurations:
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 7
LEARNING_RATE = 0.001

# model architecture
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x
```

2. **ResNet50 Model**
```python
# In Resnet50.ipynb
# Key configurations:
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001

# model architecture
resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = resnet50.to(DEVICE)
model.conv1.activation = nn.ReLU()
```

3. **Final Model with Visualization and GradCAM (77% accuracy)**
```python
# In Final_Model_with_Visualization_And_GradCAM - best 77%.ipynb
# Key configurations:
BATCH_SIZE = 256
EPOCHS = 50
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model architecture
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

# Generate Visualizations
class InferenceModel(nn.Module):
    def __init__(self, original_model):
        super(InferenceModel, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        x = self.original_model(x)
        return torch.softmax(x, dim=1)

best_model = BaselineCNN().to(DEVICE)
best_model.load_state_dict(torch.load('best_model.pth'))

inference_model = InferenceModel(best_model).to(DEVICE)

sample_indices = np.random.choice(len(test_dataset),30, replace=False)
sample_images = [test_dataset[i][0] for i in sample_indices]
sample_labels = [test_dataset[i][1] for i in sample_indices]

model.eval()
with torch.no_grad():
    sample_inputs = torch.stack(sample_images).to(DEVICE)
    sample_outputs = inference_model(sample_inputs)
    sample_probs = sample_outputs.cpu().numpy()

fig, axes = plt.subplots(5, 6, figsize=(15, 15))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i].permute(1, 2, 0))
    probs_str = ""
    for j, emotion in EMOTION_LABELS.items():
        probs_str += f"{emotion}: {sample_probs[i][j]:.2f}\n"
    ax.set_xlabel(probs_str)
plt.tight_layout()
plt.show()

# GradCAM heat maps
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import random

def visualize_gradcam_batch(model, dataset, device, num_images=10,
                             target_layer=None, use_rgb=False,
                             image_size=48, num_input_channels=1):
    model.eval()

    if target_layer is None:
        target_layer = model.conv3[-3]

    cam = GradCAM(model=model, target_layers=[target_layer])

    indices = random.sample(range(len(dataset)), num_images)

    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        input_image = image.cpu().numpy()
        input_image = np.transpose(input_image, (1, 2, 0))  # [C,H,W] -> [H,W,C]
        input_image = (input_image * 0.5) + 0.5
        input_image = np.clip(input_image, 0, 1)

        if not use_rgb and input_image.shape[2] == 1:
          input_image = np.repeat(input_image, 3, axis=2)

        visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
        ax = axes[i]
        ax.imshow(visualization)
        ax.axis('off')
        ax.set_title(f"Pred: {EMOTION_LABELS[predicted_class]}\nTrue: {EMOTION_LABELS[label]}")

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

visualize_gradcam_batch(
    model,
    test_dataset,
    device=DEVICE,
    num_images=20,
    target_layer=None,
    use_rgb=False
)
```

4. **Loss/Accuracy Curves**
```python
# accuracy curve
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.show()

# loss curve
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()
```

## Contribution
**Ruolong Mao**: BaselineCNN - 55%.ipynb, BaselineCNN_Improved - 62%.ipynb, Final_Model_with_Visualization_And_GradCAM - best 77%
**Yunshu Qiu**: Final_Model_with_Visualization_And_GradCAM - best 77%, Resnet50.ipynb
**Yat Long Choi**: Resnet50.ipynb, modified_resnet.ipynb
