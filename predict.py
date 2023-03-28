import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

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

#获取文件夹下的所有图片
#subDir: 是否获取子文件夹下的图片
def read_dir(dir_path, imgs_path, subDir=True):
    for f in os.walk(dir_path):
        if subDir:
            for subDir in f[1]:
                read_dir(dir_path + os.sep + subDir, imgs_path)
        for file in f[2]:
            if "." in file and os.path.splitext(file)[1] in IMG_EXTENSIONS:
                imgs_path.append(dir_path + os.sep + file)
        break

#给定文件夹路径, 预测文件夹下的所有图片
#subDir：是否预测子文件夹下的图片
#返回字典, key：图片相对路径(相对于输入文件夹路径) + 图片名称， value：预测类别id
def predict_directory(dir_path, model, device, subDir=True):
    model.to(device)
    imgs_path = []
    result_dic = {}
    read_dir(dir_path, imgs_path, subDir)
    for img_path in imgs_path:
        result_dic[img_path.replace(dir_path + os.sep, '')] = predict(img_path, model, device)
    return result_dic

model = load_checkpoint('best_model.pth')
pred_dir = False
image_path = ''
dir_path = ''
types = ['']
device = 'cpu'
if pred_dir:
    pred = predict_directory(dir_path, model, device)
    for key in pred.keys():
        #将字典中value类别id改为对应label
        pred[key] = types[pred[key]]
    print(pred)
else:
    pred = predict(image_path, model, device)
    print(types[pred])