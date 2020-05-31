#import argparse
import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# import lung_segmentation.importAndProcess as iap
import lung_segmentation.importAndProcess as iap
import models.model as model 
from models.unet_models import unet11, unet16

def save_mask(mask, out_dir, filename):
    filter = np.asarray(np.argmax(mask, axis=0))
    filter = (filter > 0).astype('uint8')
    filter = filter*255
    filter = np.stack((filter, filter, filter))
    pil = Image.fromarray(filter)
    pil = pil.save(f"{out_dir}/{filename}")

OUTDIR = '/home/dxtien/dxtien_research/COVID/CXR8_Segmentation'
IMGPATH = '/home/dxtien/dxtien_research/COVID/CXR8'
IMGPATHTXT = '/home/dxtien/dxtien_research/nmduy/chexnet/dataset/all.txt'
MODELPATH = '/home/dxtien/dxtien_research/nmduy/pytorch-lung-segmentation/lung_segmentation/unet16_100.pth'

normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
model = unet16(out_filters=3).cuda()
resize_dim = (224, 224)
convert_to = 'RGB'

transforms = Compose([Resize(resize_dim),ToTensor(),normalize])
convert_to = 'RGB'

dataset = iap.MyLungTest(IMGPATH, IMGPATHTXT, transforms, convert_to)


dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

#model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(MODELPATH))
#show = iap.visualize(dataset)

with torch.no_grad():
    for i, sample in enumerate(dataloader):
        img = torch.autograd.Variable(sample['image']).cuda()
        mask = model(img)
        # if not args.non_montgomery:
        #     show.ImageWithGround(i,True,True,save=True)

        # show.ImageWithMask(i, sample['filename'][0], mask.squeeze().cpu().numpy(), True, True, save=True)
        mask_np = mask.squeeze().cpu().numpy()
        filename = sample['filename']
        filename = filename.split('/')[-1]
        filename = filename[:-4]
        save_mask(mask_np, OUTDIR, filename=filename+'_mask.png')
