import itertools
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose


class DreamBoothDataset(Dataset):
    def __init__(self,
                 config,
                 data_root, 
                 prompt, 
                 class_prompt, 
                 class_data_root=None,
                 class_num=None,
                 size=1024,
                 repeats=1,
                 center_crop=False,):
        super().__init__()
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        self.data_root = Path(data_root)
        images = [Image.open(path) for path in list(self.data_root.iterdir())]

        self.instance_images = []
        for img in images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []
        interpolation = transforms.InterpolationMode.BILINEAR
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)

        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if config.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if config.center_crop:
                y1 = max(0, int(round((image.height - config.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - config.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (config.resolution, config.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            self.crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        original_size = self.original_sizes[index % self.num_instance_images]
        crop_top_left = self.crop_top_lefts[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["original_size"] = original_size
        example["crop_top_left"] = crop_top_left

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        original_sizes += [example["original_size"] for example in examples]
        crop_top_lefts += [example["crop_top_left"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch

if __name__ == '__main__':
    # data_set = DreamBoothDataset(config, './image/', None, None)
    for path in list(Path('./image/').iterdir()):
        print(path)