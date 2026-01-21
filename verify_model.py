import torch
from vas.models import SiameseResNet

def test_model():
    num_classes = 5
    model = SiameseResNet(num_classes=num_classes, backbone="resnet18")
    
    batch_size = 2
    img_size = (224, 224)
    anchor = torch.randn(batch_size, 3, *img_size)
    target = torch.randn(batch_size, 3, *img_size)
    
    output = model(anchor, target)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_classes)
    print("Verification successful!")

if __name__ == "__main__":
    test_model()
