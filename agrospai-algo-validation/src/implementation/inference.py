import torch
import os
import traceback
import numpy as np
from PIL import Image
from logging import getLogger
from torchvision.models import segmentation
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

logger = getLogger(__name__)

SOURCE_VOLUME  = '/workspace'
RECOVER_VOLUME = '/data/outputs'
ERROR_LOG_PATH = os.path.join(SOURCE_VOLUME, 'error_log.txt')

transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_model(weight_path: str, device: str):
    model = segmentation.deeplabv3_resnet101(weights=None)
    in_features = model.classifier[4].in_channels
    model.classifier[4] = torch.nn.Conv2d(in_features, 1, kernel_size=1)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
    model.to(device).eval()
    return model

def predict_and_save(model, image_path: str, output_dir: str, device: str):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)['out']
        output = torch.sigmoid(output)
        mask = (output > 0.5).float().squeeze().cpu().numpy()

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f'{basename}.png')
    mask_img.save(out_path)

def main():
    try:
        image_dir = SOURCE_VOLUME
        output_dir = os.path.join(RECOVER_VOLUME, 'predict')

        os.makedirs(SOURCE_VOLUME, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model('/algorithm/deeplabv3_apples.pth', device)

        logger.info(f'Found a total of {len(os.listdir(image_dir))} elements.')
        logger.info(f'Sample of elements found: {os.listdir(image_dir)[:5]}')
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, filename)
                predict_and_save(model, image_path, output_dir, device)

    except Exception as e:
        with open(ERROR_LOG_PATH, 'w') as f:
            f.write("❌ Error during inference:\n")
            traceback.print_exc(file=f)


if __name__ == '__main__':
    main()