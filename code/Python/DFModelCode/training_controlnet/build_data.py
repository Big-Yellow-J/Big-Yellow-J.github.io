import os
import random
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms


def build_train_dataset(config, tokenizer, accelerator):
    '''构建数据集'''
    def tokenize_captions(examples, is_train=True):
        '''将文本进行tokenizer编码'''
        captions = []
        for caption in examples[config.column_text]:
            if random.random() < config.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, 
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
        
    # 下载数据
    dataset = load_dataset(config.dataset_name,
                           cache_dir= config.cache_dir)

    # 格式化数据
    image_transforms = transforms.Compose(
        [
            transforms.Resize(config.resolution, 
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(config.resolution, 
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution),
            transforms.ToTensor(),
        ]
    )
    def preprocess_train(examples):
        '''图像处理'''
        images = [image.convert("RGB") for image in examples[config.column_image]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[config.column_conditioning_image]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if config.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=config.seed).select(range(config.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] 
                                             for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }
if __name__ == '__main__':
    build_train_dataset()