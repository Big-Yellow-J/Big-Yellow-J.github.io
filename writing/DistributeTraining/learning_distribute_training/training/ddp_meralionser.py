import os
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import warnings
import subprocess
import logging
from collections import Counter
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModelForAudioClassification

try:
    from learning_distribute_training.torchDDP_training import DDPTrainer
    from learning_distribute_training.torchDDP_config import BasicConfig
except ModuleNotFoundError:
    from torchDDP_training import DDPTrainer
    from torchDDP_config import BasicConfig
import warnings

warnings.filterwarnings("ignore", message=".*use_reentrant parameter.*")

logger = logging.getLogger(__name__)

EMO_MAP = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]
LABEL_TO_ID = {emo.lower(): i for i, emo in enumerate(EMO_MAP)}
ID_TO_LABEL = {i: emo.lower() for i, emo in enumerate(EMO_MAP)}
NUM_EMOTIONS = len(EMO_MAP)

def _focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, labels, weight=alpha, reduction="none")
    p_t = torch.exp(-ce_loss)  # 正确类别的预测概率
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce_loss).mean()

def _ccc_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Concordance Correlation Coefficient Loss (CCC Loss).
    
    用于维度情感识别 (valence/arousal/dominance 连续值)。
    CCC 衡量预测值与真实值之间的一致性，范围为 [-1, 1]，1 表示完全一致。
    loss = 1 - CCC，越接近 0 越好。
    
    Args:
        y_pred: (B, D) 预测值，D 为维度数（通常为 3: valence, arousal, dominance）
        y_true: (B, D) 真实值
    """
    mu_pred = y_pred.mean(dim=0)
    mu_true = y_true.mean(dim=0)
    var_pred = y_pred.var(dim=0, unbiased=False)
    var_true = y_true.var(dim=0, unbiased=False)
    cov = ((y_pred - mu_pred) * (y_true - mu_true)).mean(dim=0)
    
    numerator = 2.0 * cov
    denominator = var_pred + var_true + (mu_pred - mu_true).pow(2)
    ccc = numerator / (denominator + 1e-8)
    return (1.0 - ccc).mean()

def _load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ac", "1", "-ar", str(target_sr),
        "-f", "f32le", "-loglevel", "error", "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True).stdout
    if not raw:
        raise RuntimeError(f"ffmpeg 无法解码: {audio_path}")
    return np.frombuffer(raw, dtype=np.float32)

@dataclass
class MeralionSERConfig(BasicConfig):
    task_type: str = "classification"
    project_name: str = "MeralionSER-Training-Mdiri"
    model_name_or_path: str = "MERaLiON/MERaLiON-SER-v1"
    store_dir: str = "/home/huangjie/MdiriCode/ModelTrainingResult"
    cache_dir: str = "/home/huangjie/MdiriCode/ModelParameterCache"

    audio_path_dir: str = "/home/huangjie/MdiriCode/SER/data/studio"
    data_path_list: List[str] = field(
        default_factory=lambda: ["/home/huangjie/MdiriCode/SER/SER/data/train.csv", 
                                 "/home/huangjie/MdiriCode/SER/SER/data/val.csv",
                                 "/home/huangjie/MdiriCode/SER/SER/data/allin_ser_dataset_review.xlsx"])

    resample_target_count: int = 200
    resample_labels: List[str] = field(
        default_factory=lambda: ["sad", "disgust", "fear", "surprised"])  # 需要重采样的标签
    num_proc: int = 1
    epoch: int = 20
    batch_size: int = 192 # 12   
    learning_rate: float = 6e-4
    checkpointing_steps: int = -1
    checkpoints_total_limit: int = 2
    seed: int = 42
    max_train_steps: int = 0

    best_metric_name: str = "Eval/uar"
    best_metric_mode: str = "max"

    loss_type: str = "cross_entropy"
    focal_gamma: float = 2.0             # Focal Loss 的 gamma 参数（仅 loss_type=focal 时生效）
    focal_alpha: Optional[float] = None  # Focal Loss 的 alpha（类别权重），None 则不使用
    ccc_loss_weight: float = 0.0         # CCC loss 权重（模型输出 dims 用于 valence/arousal/dominance 回归）
    label_smoothing: float = 0.1         # Cross Entropy 的 label smoothing

    # 微调策略: "lora_only" (只训 LoRA 层) | "lora_plus_downstream" (LoRA + downstream_model 全量微调) | "lora_plus_classifier" (LoRA + 情绪分类头微调) | "classifier_only" (仅微调 emotion_classification_layer)
    finetune_strategy: str = "classifier_only"# "lora_only"
    # DDP find_unused_parameters: classifier_only 下关闭可避免性能警告，其他策略建议开启
    ddp_find_unused_params: bool = False

    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.03
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 2
    log_with: str = "tensorboard"
    optim_name: str = "adamw"
    best_metric_name: str = "Eval/uar"
    best_metric_mode: str = "max"
    gradient_checkpointing_kwargs: Optional[dict] = None
    gradient_checkpointing: bool = False  # MERaLiON-SER 不支持 gradient checkpointing
    distributed_strategy: str = "ddp"

class MeralionSERTrainer(DDPTrainer):
    def __init__(self, config: MeralionSERConfig):
        super().__init__(config=config, model=None,
                         train_dataset=None, eval_dataset=None)
        model, processor = self._load_model(config)
        self.processor = processor
        self.model = model
        train_dataset, eval_dataset = self._load_dataset(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._build_dataloader()
        self._prepare_model_optimizer_scheduler()

    def _collate_fn(self, batch: list) -> dict:
        paths = [item["path"] for item in batch]
        labels = [item["label"] for item in batch]

        wavs = [_load_audio(p) for p in paths]
        label_ids = [LABEL_TO_ID[lbl.strip().lower()] for lbl in labels]

        inputs = self.processor(
            wavs, sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            max_length=16000 * 30,
        )
        inputs["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return inputs

    def _load_model(self, config: MeralionSERConfig):
        snapshots_dir = os.path.join(
            config.cache_dir,
            "models--MERaLiON--MERaLiON-SER-v1", "snapshots"
        )
        local_path = None
        if os.path.isdir(snapshots_dir):
            dirs = sorted(os.listdir(snapshots_dir))
            for d in dirs:
                candidate = os.path.join(snapshots_dir, d)
                if os.path.isdir(candidate) and os.path.isfile(
                    os.path.join(candidate, "modeling_ser_whisper_ecapa.py")
                ):
                    local_path = candidate
                    break

        # 加载前修复 snapshot 中的 torch.logspace → meta tensor 问题
        if local_path:
            self._patch_modeling_file(local_path)

        repo = local_path if local_path else config.model_name_or_path
        self.log("info", "[MERaLiON-SER] Loading from: %s", repo)

        processor = AutoProcessor.from_pretrained(
            repo, cache_dir=config.cache_dir,
            trust_remote_code=True,
            local_files_only=local_path is not None,
        )
        model = AutoModelForAudioClassification.from_pretrained(
            repo, cache_dir=config.cache_dir,
            trust_remote_code=True,
            local_files_only=local_path is not None,
        )
        # 1：64 128
        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            task_type="SEQ_CLS",
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)

        finetune_strategy = getattr(config, "finetune_strategy", "lora_only")
        if finetune_strategy == "classifier_only":
            # 仅微调情绪分类头，不注入 LoRA，冻结其它所有参数
            for n, p in model.named_parameters():
                p.requires_grad = "emotion_classification_layer" in n
        else:
            model = get_peft_model(model, peft_config)

            if finetune_strategy == "lora_only":
                for n, p in model.named_parameters():
                    p.requires_grad = "lora_" in n
            elif finetune_strategy == "lora_plus_downstream":
                for n, p in model.named_parameters():
                    if "downstream_model" in n or "emotion_layer" in n or "dim_layer" in n or "emotion_classification_layer" in n:
                        p.requires_grad = True
                    elif "lora_" in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            elif finetune_strategy == "lora_plus_classifier":
                for n, p in model.named_parameters():
                    if "emotion_classification_layer" in n:
                        p.requires_grad = True
                    elif "lora_" in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                for n, p in model.named_parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.log(
            "info",
            "[MERaLiON-SER] Finetune strategy=%s, LoRA applied. Trainable: %s / %s (%.2f%%)",
            finetune_strategy, f"{trainable:,}", f"{total:,}", 100 * trainable / total,
        )
        return model, processor

    def _patch_modeling_file(self, snapshot_dir: str) -> None:
        modeling_file = os.path.join(snapshot_dir, "modeling_ser_whisper_ecapa.py")
        if not os.path.isfile(modeling_file):
            return
        with open(modeling_file, "r") as f:
            content = f.read()

        patched = False

        # Fix 1: torch.logspace → pure Python
        old_logspace = (
            "        # Generate kernel sizes in log space\n"
            "        kernel_sizes = [int(k) for k in torch.logspace(\n"
            "            math.log10(min_kernel), math.log10(max_kernel), num_resolutions\n"
            "        )]"
        )
        if old_logspace in content:
            new_logspace = (
                "        # Generate kernel sizes in log space (patched: pure Python, avoids meta tensor .item() error)\n"
                "        kernel_sizes = [\n"
                "            int(10 ** (math.log10(min_kernel) + i * (math.log10(max_kernel) - math.log10(min_kernel)) / max(1, num_resolutions - 1)))\n"
                "            for i in range(num_resolutions)\n"
                "        ]"
            )
            content = content.replace(old_logspace, new_logspace)
            patched = True

        # Fix 2: remove @torch.no_grad() from forward (model was inference-only)
        old_nograd = "    @torch.no_grad()\n    def forward(self, input_values=None, sampling_rate=None, input_features=None, attention_mask=None, **kwargs):"
        if old_nograd in content:
            new_nograd = "    # PATCHED: removed @torch.no_grad() to enable training (original model was inference-only)\n    def forward(self, input_values=None, sampling_rate=None, input_features=None, attention_mask=None, **kwargs):"
            content = content.replace(old_nograd, new_nograd)
            patched = True

        if patched:
            with open(modeling_file, "w") as f:
                f.write(content)
            self.log(
                "info",
                "[MERaLiON-SER] Patched %s — removed @torch.no_grad() + fixed torch.logspace.",
                modeling_file,
            )

    def _concat_all_data(self, config: MeralionSERConfig):
        all_paths, all_labels = [], []
        audio_dir = config.audio_path_dir

        for data_path in config.data_path_list:
            try:
                df = pd.read_csv(data_path)
            except Exception:
                df = pd.read_excel(data_path)

            for _, row in df.iterrows():
                # 兼容 csv (audio_key) 和 xlsx (sample_id) 列名
                audio_name = row.get("audio_key", row.get("sample_id", None))
                audio_label = row.get("label", None)
                if audio_name is None or audio_label is None:
                    continue

                # 查找实际音频文件
                audio_file = None
                candidate = os.path.join(audio_dir, str(audio_name))
                if os.path.isfile(candidate):
                    audio_file = candidate
                else:
                    for ext in [".ogg", ".m4a", ".mp3", ".wav"]:
                        p = os.path.join(audio_dir, f"{audio_name}{ext}")
                        if os.path.isfile(p):
                            audio_file = p
                            break

                if audio_file is None:
                    continue

                label_str = str(audio_label).strip().lower()
                if label_str not in LABEL_TO_ID:
                    continue

                all_paths.append(audio_file)
                all_labels.append(label_str)

        return all_paths, all_labels

    def _resample_minority(
        self, paths, labels, target_count: int, minority_labels: list, seed: int = 42
    ):
        rng = np.random.RandomState(seed)
        resampled_paths = list(paths)
        resampled_labels = list(labels)

        for lbl in minority_labels:
            indices = [i for i, l in enumerate(labels) if l == lbl]
            current = len(indices)
            if current <= 0 or current >= target_count:
                continue
            needed = target_count - current
            extra_idx = rng.choice(indices, size=needed, replace=True).tolist()
            for idx in extra_idx:
                resampled_paths.append(paths[idx])
                resampled_labels.append(labels[idx])

        return resampled_paths, resampled_labels

    def _load_dataset(self, config: MeralionSERConfig):
        all_paths, all_labels = self._concat_all_data(config)
        self.log("info", "[Data] Total valid samples: %s", len(all_paths))

        self.log("info", "[Data] Label distribution: %s", dict(Counter(all_labels)))

        train_p, eval_p, train_l, eval_l = train_test_split(
            all_paths, all_labels,
            test_size=0.2,
            random_state=config.seed,
            stratify=all_labels,
        )

        if config.resample_target_count and config.resample_labels:
            train_p, train_l = self._resample_minority(
                train_p, train_l,
                target_count=config.resample_target_count,
                minority_labels=config.resample_labels,
                seed=config.seed,
            )

            self.log("info", "[Data] After resample train labels: %s", dict(Counter(train_l)))

        self.log("info", "[Data] Train: %s, Eval: %s", len(train_p), len(eval_p))

        train_dataset = Dataset.from_dict({"path": train_p, "label": train_l})
        eval_dataset = Dataset.from_dict({"path": eval_p, "label": eval_l})
        return train_dataset, eval_dataset

    def _build_dataloader(self) -> None:
        super()._build_dataloader()
        for attr in ("train_dataloader", "eval_dataloader"):
            loader = getattr(self, attr, None)
            if loader is not None:
                setattr(self, attr,
                    DataLoader(
                        loader.dataset,
                        batch_size=loader.batch_size,
                        sampler=loader.sampler,
                        shuffle=False,
                        num_workers=loader.num_workers,
                        pin_memory=loader.pin_memory,
                        drop_last=loader.drop_last,
                        persistent_workers=getattr(loader, "persistent_workers", False),
                        prefetch_factor=getattr(loader, "prefetch_factor", None),
                        collate_fn=self._collate_fn,
                    ))
                
    def compute_loss(self, batch) -> torch.Tensor:
        batch = self._move_batch_to_device(batch)
        labels = batch.pop("labels", None)
        dim_labels = batch.pop("dims", None)  # (B, 3) valence/arousal/dominance 标签，可选
        if labels is None:
            raise ValueError("Batch missing 'labels' key")

        outputs = self.model(
            input_features=batch.get("input_features"),
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        logits = outputs["logits"]  # (B, num_emotions)

        # ---- 分类 loss ----
        loss_type = getattr(self.config, "loss_type", "cross_entropy")
        label_smoothing = getattr(self.config, "label_smoothing", 0.0)
        if loss_type == "focal":
            gamma = getattr(self.config, "focal_gamma", 2.0)
            alpha_val = getattr(self.config, "focal_alpha", None)
            alpha = torch.tensor(alpha_val, device=labels.device, dtype=torch.float) if alpha_val is not None else None
            cls_loss = _focal_loss(logits, labels, gamma=gamma, alpha=alpha)
        else:
            cls_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        # ---- CCC loss（维度情感回归，如果有 dim_labels） ----
        ccc_weight = getattr(self.config, "ccc_loss_weight", 0.0)
        dims = outputs.get("dims", None)
        if ccc_weight > 0 and dims is not None and dim_labels is not None:
            ccc_loss = _ccc_loss(dims, dim_labels)
            return cls_loss + ccc_weight * ccc_loss

        return cls_loss
    
    def evaluate(self) -> dict:
        from tqdm import tqdm as _tqdm
        from sklearn.metrics import recall_score, f1_score, accuracy_score
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            pbar = _tqdm(
                total=len(self.eval_dataloader),
                disable=not self.is_main_process,
                desc="EVAL", dynamic_ncols=True,
            )
            for batch in self.eval_dataloader:
                batch = self._move_batch_to_device(batch)
                labels = batch.pop("labels", None)

                outputs = self.model(
                    input_features=batch.get("input_features"),
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
                logits = outputs["logits"]
                loss = F.cross_entropy(logits, labels)
                total_loss += float(loss.detach().item())

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                pbar.update(1)

        self.model.train()

        # 多卡汇总：先收集各卡的结果到 rank 0，再统一计算指标
        if self.is_distributed:
            gathered_preds = [None for _ in range(self.world_size)]
            gathered_labels = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_preds, all_preds)
            dist.all_gather_object(gathered_labels, all_labels)
            if self.is_main_process:
                all_preds = [p for sub in gathered_preds for p in sub]
                all_labels = [l for sub in gathered_labels for l in sub]

        # loss 汇总
        loss_tensor = torch.tensor(total_loss, device=self.device)
        loss_tensor = self._reduce_scalar(loss_tensor)

        metrics = {
            "Eval/eval_loss": round(float(loss_tensor.item()) / max(1, len(self.eval_dataloader)), 4),
        }

        if self.is_main_process and len(all_labels) > 0:
            acc = accuracy_score(all_labels, all_preds)
            uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            metrics["Eval/accuracy"] = round(float(acc), 4)
            metrics["Eval/uar"] = round(float(uar), 4)
            metrics["Eval/f1_macro"] = round(float(f1_macro), 4)

            # ---- 逐类指标：Recall, Precision, F1 ----
            from sklearn.metrics import precision_score
            per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
            per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
            per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
            for i, emo in enumerate(EMO_MAP):
                emo_lower = emo.lower()
                metrics[f"Eval/{emo_lower}_precision"] = round(float(per_class_precision[i]) if i < len(per_class_precision) else 0.0, 4)
                metrics[f"Eval/{emo_lower}_recall"] = round(float(per_class_recall[i]) if i < len(per_class_recall) else 0.0, 4)
                metrics[f"Eval/{emo_lower}_f1"] = round(float(per_class_f1[i]) if i < len(per_class_f1) else 0.0, 4)
        return metrics

    # ---------- 推理 & 测试 ----------

    def load_model_for_inference(self, checkpoint_path: str, device: str = "cuda"):
        """加载微调后的模型权重用于推理/测试。

        Args:
            checkpoint_path: training_state.pt 所在目录路径（如 .../checkpoint-best-200）
            device: 推理设备，默认 "cuda"

        Returns:
            model, processor — 可直接用于推理
        """
        state_file = os.path.join(checkpoint_path, "training_state.pt")
        if not os.path.isfile(state_file):
            raise FileNotFoundError(f"Checkpoint not found: {state_file}")

        state = torch.load(state_file, map_location="cpu", weights_only=False)
        saved_config = state.get("config", {})
        finetune_strategy = saved_config.get("finetune_strategy", "lora_only")

        self.log("info", "[Inference] Loading model from: %s", checkpoint_path)
        self.log("info", "[Inference] finetune_strategy=%s", finetune_strategy)

        # 1. 重建原始模型（与训练时一致的结构）
        processor = self.processor  # 复用 trainer 已有的 processor
        repo = saved_config.get("model_name_or_path", "MERaLiON/MERaLiON-SER-v1")

        # 重新加载基础模型
        model = AutoModelForAudioClassification.from_pretrained(
            repo,
            cache_dir=saved_config.get("cache_dir"),
            trust_remote_code=True,
            local_files_only=True,
        )

        # 2. 按训练时的策略注入 LoRA（如果需要）
        if finetune_strategy != "classifier_only":
            peft_config = LoraConfig(
                r=saved_config.get("lora_r", 32),
                lora_alpha=saved_config.get("lora_alpha", 64),
                lora_dropout=saved_config.get("lora_dropout", 0.05),
                task_type="SEQ_CLS",
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model = get_peft_model(model, peft_config)

        # 3. 加载训练好的权重
        model.load_state_dict(state["model"], strict=False)
        model.to(device)
        model.eval()

        self.log("info", "[Inference] Model loaded successfully, device=%s", device)
        return model, processor

    def evaluate_with_confusion_matrix(
        self, model=None, processor=None, data_path_list: Optional[List[str]] = None,
        audio_path_dir: Optional[str] = None, device: str = "cuda",
    ) -> dict:
        """单卡评估：加载数据 → 推理 → 计算所有指标 + 混淆矩阵。

        如果 model/processor 为 None，则使用 self.model / self.processor（训练中的评估）。
        否则可以使用 load_model_for_inference() 加载的模型进行独立测试。

        Returns:
            dict: {
                "accuracy": float,
                "uar": float,
                "f1_macro": float,
                "confusion_matrix": np.ndarray (C, C),
                "per_class_precision": list[float],
                "per_class_recall": list[float],
                "per_class_f1": list[float],
                "preds": list[int],
                "labels": list[int],
            }
        """
        from tqdm import tqdm as _tqdm
        from sklearn.metrics import (
            recall_score, f1_score, accuracy_score,
            precision_score, confusion_matrix,
        )

        model = model or self.model
        processor = processor or self.processor
        model.eval()

        # 如果没有传入数据路径，复用 trainer 的 eval_dataset 对应的数据
        if data_path_list is None:
            # 从 self.eval_dataset 中提取所有数据
            all_paths = [item["path"] for item in self.eval_dataset]
            all_label_strs = [item["label"] for item in self.eval_dataset]
        else:
            # 从外部文件加载数据
            audio_dir = audio_path_dir or self.config.audio_path_dir
            config = MeralionSERConfig(
                audio_path_dir=audio_dir,
                data_path_list=data_path_list,
            )
            # 创建一个临时的 trainer 实例来解析数据
            temp_trainer = MeralionSERTrainer.__new__(MeralionSERTrainer)
            all_paths, all_label_strs = temp_trainer._concat_all_data(config)

        label_ids = [LABEL_TO_ID[lbl.strip().lower()] for lbl in all_label_strs]
        self.log("info", "[EvalCM] Total samples: %s", len(all_paths))

        all_preds: List[int] = []
        all_labels: List[int] = []

        # 逐样本推理（不用 DataLoader 的 collate_fn，避免 batch 处理复杂性）
        with torch.no_grad():
            for i in _tqdm(range(len(all_paths)), desc="Inference"):
                audio = _load_audio(all_paths[i])
                inputs = processor(
                    audio, sampling_rate=16000,
                    return_tensors="pt",
                    padding="max_length",
                    return_attention_mask=True,
                    truncation=True,
                    max_length=16000 * 30,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(
                    input_features=inputs.get("input_features"),
                    attention_mask=inputs.get("attention_mask"),
                )
                logits = outputs["logits"]
                pred = int(logits.argmax(dim=-1).item())
                all_preds.append(pred)
                all_labels.append(label_ids[i])

        # 计算指标
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_EMOTIONS)))
        acc = accuracy_score(all_labels, all_preds)
        uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        results = {
            "accuracy": round(float(acc), 4),
            "uar": round(float(uar), 4),
            "f1_macro": round(float(f1_macro), 4),
            "confusion_matrix": cm,
            "per_class_precision": {EMO_MAP[i]: round(float(per_class_precision[i]), 4) for i in range(NUM_EMOTIONS)},
            "per_class_recall": {EMO_MAP[i]: round(float(per_class_recall[i]), 4) for i in range(NUM_EMOTIONS)},
            "per_class_f1": {EMO_MAP[i]: round(float(per_class_f1[i]), 4) for i in range(NUM_EMOTIONS)},
            "preds": all_preds,
            "labels": all_labels,
        }

        # 打印结果
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"UAR:       {results['uar']:.4f}")
        print(f"F1-Macro:  {results['f1_macro']:.4f}")
        print("-" * 60)
        print(f"{'Emotion':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 60)
        for emo in EMO_MAP:
            p = results["per_class_precision"][emo]
            r = results["per_class_recall"][emo]
            f = results["per_class_f1"][emo]
            print(f"{emo:<12} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
        print("-" * 60)
        print("\nConfusion Matrix (rows=true, cols=pred):")
        cm_str = "           " + "".join(f"{e[:6]:>7}" for e in EMO_MAP)
        print(cm_str)
        for i, emo in enumerate(EMO_MAP):
            row = "".join(f"{cm[i][j]:>7}" for j in range(NUM_EMOTIONS))
            print(f"{emo:<10}: {row}")
        print("=" * 60 + "\n")

        return results

if __name__ == "__main__":
    #export HF_ENDPOINT=https://hf-mirror.com && CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 ddp_meralionser.py
    config = MeralionSERConfig()
    trainer = MeralionSERTrainer(config)
    trainer.train()

    model, processor = trainer.load_model_for_inference("/home/huangjie/MdiriCode/CodeLearning/ModelTrainingResult/20260526-MeralionSER-Training-Mdiri-1573-ddp/checkpoint-best-49/")
    results = trainer.evaluate_with_confusion_matrix()

    # model, processor = trainer.load_model_for_inference(".../checkpoint-best-200")
    # results = trainer.evaluate_with_confusion_matrix(
    #     model=model, processor=processor,
    #     data_path_list=["/path/to/test.csv"],
    #     audio_path_dir="/path/to/audio",
    #     device="cuda",
    # )