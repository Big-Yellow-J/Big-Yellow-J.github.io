import math
import json
import itertools
import webdataset as wds
import torchvision.transforms.functional as TF
from typing import List, Union
from torchvision import transforms
from braceexpand import braceexpand
from torch.utils.data import default_collate
from diffusers.training_utils import resolve_interpolation_mode
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

class WebdatasetFilter:
    def __init__(self, min_size=1024, max_pwatermark=0.5):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark

    def __call__(self, x):
        try:
            if "json" in x:
                x_json = json.loads(x["json"])
                filter_size = (x_json.get("original_width", 0.0) or 0.0) >= self.min_size and x_json.get(
                    "original_height", 0
                ) >= self.min_size
                filter_watermark = (x_json.get("pwatermark", 1.0) or 1.0) <= self.max_pwatermark
                return filter_size and filter_watermark
            else:
                return False
        except Exception:
            return False


class SDText2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int= 4,
        resolution: int = 512,
        interpolation_type: str = "bilinear",
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        interpolation_mode = resolve_interpolation_mode(interpolation_type)

        def transform(example):
            # resize image
            image = example["image"]
            image = TF.resize(image, resolution, interpolation=interpolation_mode)

            # get crop coordinates and crop image
            c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
            image = TF.crop(image, c_top, c_left, resolution, resolution)
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])

            example["image"] = image
            return example

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=wds.warn_and_continue),
            wds.map(filter_keys({"image", "text"})),
            wds.map(transform),
            wds.to_tuple("image", "text"),
        ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

if __name__ =='__main__':
    from LCM_config import LCMTrainingConfig
    config = LCMTrainingConfig()
    dataset = SDText2ImageDataset(train_shards_path_or_url= config.train_data_path,
                        num_train_examples= config.num_train_samples,
                        per_gpu_batch_size= config.train_batch_size,
                        global_batch_size= config.train_batch_size)
    train_dataset = dataset.train_dataloader