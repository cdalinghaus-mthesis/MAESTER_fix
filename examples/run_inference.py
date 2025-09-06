import torch
import sys
sys.path.append("../MAESTER/")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from model import *
from utils import get_plugin, read_yaml
import numpy as np
from PIL import Image, ImageSequence
from torchvision.transforms import ToTensor
from infer_engine import run_inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config="default"
cfg = read_yaml(f"../MAESTER/config/{config}.yaml")
model_weights = torch.load(f"../MAESTER/model_weights/{config}/model.pt", map_location="cpu")
#kmeans_center = np.loadtxt(f"../MAESTER/model_weights/{config}/k6.txt")
storage_path = "./temp.pth" # path to save the intermediate results
K_CENTRE = 2
dataset_name = "c4"
data_path = f"../MAESTER/data/download/high_{dataset_name}/high_{dataset_name}_source.tif" #


model = get_plugin('model', cfg["MODEL"]["name"])(cfg["MODEL"])
model.load_state_dict(model_weights)
model.to(device)


vol_size = cfg["DATASET"]["vol_size"] + int(cfg["DATASET"]["vol_size"] // cfg["DATASET"]["patch_size"])  if cfg["DATASET"]["patch_size"] % 2 == 0 else cfg["DATASET"]["vol_size"]
TAR_SIZE = cfg["DATASET"]["vol_size"]
PATCH_size = cfg["DATASET"]["patch_size"]
OFFSET = PATCH_size // 2
PIX_SIZE = cfg["DATASET"]["patch_size"] + 1 if cfg["DATASET"]["patch_size"] % 2 == 0 else cfg["DATASET"]["patch_size"]
CENTRAL_PATCH = 4
PAD = int(OFFSET + ((((TAR_SIZE / PATCH_size) - CENTRAL_PATCH) // 2) * (PATCH_size + 1 if PATCH_size % 2==0 else PATCH_size)))
PAD = PAD + PIX_SIZE // 2

"""
image = Image.open(data_path)
tt = ToTensor()
count = 0
tensor_stack = []
for i, page in enumerate(ImageSequence.Iterator(image)):
    array = tt(page)
    tensor_stack.append(array)
    count += 1
result = torch.concat(tensor_stack, dim=0)
orgD, orgH, orgW = result.shape
"""

result = torch.load('../data/dataset/Fluo-N2DL-HeLa_02_source_tensor')['Fluo-N2DL-HeLa_02_source'][0]
orgD, orgH, orgW = result.shape
print(result.shape)


# select two slices for inference
START, END= 10, 12
src = result[START:END]
rank=0
ngpus_per_node=1
run_inference(rank, ngpus_per_node, src, cfg, model, storage_path, device)


feature_storage = torch.FloatTensor(torch.FloatStorage.from_file(storage_path, shared=True, size=orgD * (orgH+PAD) * (orgW+PAD) * cfg["MODEL"]["embed"])).reshape(orgD, orgH+PAD, orgW+PAD, cfg["MODEL"]["embed"])
feature_flattern = feature_storage[0, :orgH,:orgW,: ].flatten(0, -2).clone().numpy().astype(float)

K_CENTRE = 2
kmeans = KMeans(n_clusters=K_CENTRE, random_state=2)
kmeans.fit(feature_flattern)
#kmeans.cluster_centers_ = kmeans_center
kmeans._n_threads = 1
seg_pred = kmeans.predict(feature_flattern)
seg_pred[seg_pred == 0] = 4  # class merge


#plt.imshow(np.array(src[0].cpu()), interpolation='bilinear')
#plt.imshow(np.array(seg_pred.reshape(1, orgH , -1)[0]), alpha=0.5, 
#           interpolation='bilinear')
#plt.savefig("fig.png")

plt.figure(figsize=(12, 8))  # width=12 inches, height=8 inches

plt.imshow(np.array(src[0].cpu()), interpolation='bilinear')
plt.imshow(np.array(seg_pred.reshape(1, orgH, -1)[0]), 
           alpha=0.5, interpolation='bilinear')

plt.savefig("figlarger.png", dpi=300)  # optional: increase dpi for better quality
plt.show()
