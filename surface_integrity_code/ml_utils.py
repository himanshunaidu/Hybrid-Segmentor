import torch
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "..", "Hybrid_Segmentor"))
from model import HybridSegmentor
from utils import save_predictions_as_imgs, eval_metrics, eval_ODS, eval_OIS
from dataloader import get_loaders, get_loader
import config

mu = [0.51789941, 0.51360926, 0.547762]
sd  = [0.1812099,  0.17746663, 0.20386334]

def get_model():
    model = HybridSegmentor().to(config.DEVICE)
    
    ck_file_path = r'/home/hnaidu36/Desktop/SurfaceIntegrity/Hybrid-Segmentor/hybrid_segmentor_BCE_2.ckpt'
    checkpoint = torch.load(ck_file_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model

def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        x = image_tensor.to(config.DEVICE).unsqueeze(0)  # Add batch dimension
        x = (x - torch.tensor(mu).view(1, 3, 1, 1).to(config.DEVICE)) / torch.tensor(sd).view(1, 3, 1, 1).to(config.DEVICE)
        output = model(x)[0]
        preds_probability = torch.sigmoid(output)
        preds = (preds_probability > 0.5).float()
    return preds.squeeze(0).cpu()[0]