import os
import glob
import cv2
import random
import torch
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets

from mask_generate import FaceMaskGenerator

def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img

class CustomDataset(Dataset):
    def __init__(self, hugging_name=['saitsharipov/CelebA-HQ', 'nielsr/CelebA-faces'], 
                 cache_dir='/data/huangjie', mask='custom', 
                 image_size=512, dataset_type='train'):
        super().__init__()
        self.mask_mode = mask
        self.image_size = image_size
        self.dataset_type = dataset_type

        if isinstance(hugging_name, list):
            datasets_list = [load_dataset(name, split='train', cache_dir=cache_dir, streaming= False) 
                             for name in hugging_name]
            self.dataset = concatenate_datasets(datasets_list)
        else:
            self.dataset = load_dataset(hugging_name, split= 'train', cache_dir=cache_dir, streaming=False)
   
        train_len = int(len(self.dataset) * 0.95)
        if self.dataset_type == 'train':
            self.dataset = self.dataset.select(range(train_len))
        elif self.dataset_type == 'test':
            self.dataset = self.dataset.select(range(train_len, len(self.dataset)))

        self.gen = FaceMaskGenerator(mode= random.choice(['freeform']), image_size= self.image_size,
                                     target_ratio= random.choice([0.5, 0.8, 0.2]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.

        mask = torch.from_numpy(self.gen()).unsqueeze(0).float()
        masked_img = image * (1 - mask)
        input_tensor = torch.cat([masked_img, mask], dim=0)

        # masked_img_pil = Image.fromarray((masked_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        # masked_img_pil.save("./masked.png")
        return {"gt": image, "mask": mask, "input": input_tensor}

class InpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.jpg'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        result = dict(image=image, mask=mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    dataset = CustomDataset(dataset_type= 'test')
    trian_dataloader = DataLoader(dataset, 32)

    s_time = time.time()
    with tqdm(total= len(trian_dataloader)) as par:
        for i, batch in enumerate(trian_dataloader):
            par.update(1)
    print(time.time()- s_time)