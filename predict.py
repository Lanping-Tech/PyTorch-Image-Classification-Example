import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def load_checkpoint(checkpoint_path):
    model = torch.load(checkpoint_path, map_location='cpu') 
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def process_image(image):
    mean = [0.5, 0.5, 0.5]
    std = [0.229, 0.224, 0.225]
    pil_im = Image.open(f'{image}')
    transformations = transforms.Compose([transforms.Resize(320),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,
                                                           std)])

    pil_transformed = transformations(pil_im)
    image_data = np.array(pil_transformed)
    return image_data
    
def predict(image_path, model, device):
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = torch.from_numpy(img_torch).unsqueeze_(0)
    img_torch = img_torch.float()
    model.eval()
    with torch.no_grad():
        output = model(img_torch)
        model.train()

    pred = F.softmax(output, dim=1)
    pred = torch.max(pred.data, 1)
    pred = pred[1].item()
    return pred

model = load_checkpoint('best_model.pth')
image_path = ''
pred = predict(image_path, model, 'cpu')
types = ['']
print(types[pred])