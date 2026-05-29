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
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoModelForAudioClassification

try:
    from learning_distribute_training.torchDDP_training import DDPTrainer
    from learning_distribute_training.torchDDP_config import BasicConfig
except ModuleNotFoundError:
    from torchDDP_training import DDPTrainer
    from torchDDP_config import BasicConfig

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
    mu_pred = y_pred.mean(dim=0)
    mu_true = y_true.mean(dim=0)
    var_pred = y_pred.var(dim=0, unbiased=False)
    var_true = y_true.var(dim=0, unbiased=False)
    cov = ((y_pred - mu_pred) * (y_true - mu_true)).mean(dim=0)
    
    numerator = 2.0 * cov
    denominator = var_pred + var_true + (mu_pred - mu_true).pow(2)
    ccc = numerator / (denominator + 1e-8)
    return (1.0 - ccc).mean()

def _load_audio(audio_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ac", "1", "-ar", str(target_sr),
        "-f", "f32le", "-loglevel", "error", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    raw = result.stdout
    if not raw:
        stderr_msg = result.stderr.decode(errors="replace").strip()[:200] if result.stderr else "no stderr"
        logger.warning(f"ffmpeg 无法解码: {audio_path} | {stderr_msg}")
        return None
    return np.frombuffer(raw, dtype=np.float32)

@dataclass
class MeralionSERConfig(BasicConfig):
    task_type: str = "classification"
    project_name: str = "Mdiri-Extra"
    model_name_or_path: str = "MERaLiON/MERaLiON-SER-v1"
    store_dir: str = "/home/huangjie/MdiriCode/ModelTrainingResult"
    cache_dir: str = "/home/huangjie/MdiriCode/ModelParameterCache"
    audio_path_dir: str = "/home/huangjie/MdiriCode/SER/data/studio"
    data_path_list: List[str] = field(
        default_factory=lambda: [
            "/home/huangjie/MdiriCode/SER/SER/train.csv",
        ])
    eval_data_path_list: List[str] = field(
        default_factory=lambda: [
            "/home/huangjie/MdiriCode/SER/SER/test.csv",
        ])

    resample_target_count: int = None
    resample_labels: List[str] = field(default_factory=lambda: ["sad", "disgusted", "fearful", "surprised"])
    neutral_max_samples: int = 0  # neutral 最大保留数，0=不限制；推荐 4000~6000（配合 resample 可设更小）
    num_proc: int = 1
    epoch: int = 60
    max_length: int = 16000 * 30 # Whisper 编码器要求 mel 特征固定 3000 帧（= 30秒@16kHz)
    batch_size: int = 64
    learning_rate: float = 1e-5
    checkpointing_steps: int = -1
    checkpoints_total_limit: int = 2
    seed: int = 42
    max_train_steps: int = 0

    loss_type: str = "cross_entropy" #"focal"
    focal_gamma: float = 3.0             # Focal Loss 的 gamma，越大越关注难样本（不均衡严重时建议 3~5）
    # alpha: 平方根逆频率权重 (1/sqrt(count) 归一化)，对应 [neutral, happy, sad, angry, fearful, disgusted, surprised]
    focal_alpha: Optional[List[float]] = field(
        default_factory=lambda: [0.0229, 0.1025, 0.2108, 0.0815, 0.2391, 0.1930, 0.1502]
    )
    ccc_loss_weight: float = 0.0         # CCC loss 权重（模型输出 dims 用于 valence/arousal/dominance 回归）
    label_smoothing: float = 0.0         # Cross Entropy 的 label smoothing

    # "lora_only" (只训 LoRA 层) 
    # "lora_plus_downstream" (LoRA + downstream_model 全量微调) 
    # "lora_plus_classifier" (LoRA + 情绪分类头微调) 
    # "classifier_only" (仅微调 emotion_classification_layer，使用模型自带分类头)
    finetune_strategy: str = "classifier_only"
    ddp_find_unused_params: bool = False

    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.1
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    log_with: str = "tensorboard"
    optim_name: str = "adamw"
    best_metric_name: str = "Eval/uar"
    best_metric_mode: str = "max"
    gradient_checkpointing_kwargs: Optional[dict] = None
    gradient_checkpointing: bool = False  # MERaLiON-SER 不支持 gradient checkpointing
    distributed_strategy: str = "ddp"

    def _concat_all_data(self, config):
        return _concat_all_data_static(config.data_path_list)

def _concat_all_data_static(data_path_list: List[str]):
    """从 CSV 文件读取音频路径和标签。支持 'audio_path' / 'path' 列。"""
    all_paths, all_labels = [], []

    for data_path in data_path_list:
        try:
            df = pd.read_csv(data_path)
        except Exception:
            df = pd.read_excel(data_path)

        for _, row in df.iterrows():
            # 优先 audio_path 列，回退到 path 列
            audio_file = row.get("audio_path", None)
            if pd.isna(audio_file) or not str(audio_file).strip():
                audio_file = row.get("path", None)
            if pd.isna(audio_file) or not str(audio_file).strip():
                continue
            audio_file = str(audio_file).strip()

            audio_label = row.get("label", None)
            if pd.isna(audio_label) or not str(audio_label).strip():
                continue

            label_str = str(audio_label).strip().lower()
            if label_str not in LABEL_TO_ID:
                continue

            all_paths.append(audio_file)
            all_labels.append(label_str)
    return all_paths, all_labels

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
        """过滤掉无法加载的音频，避免 ffmpeg 解码失败导致训练卡住。"""
        valid_items = []
        for item in batch:
            audio = _load_audio(item["path"])
            if audio is not None:
                valid_items.append((audio, item["label"]))

        if not valid_items:
            # 极端情况：整批都无法加载，返回一个空 batch（几乎不会发生）
            inputs = self.processor(
                [np.zeros(16000, dtype=np.float32)], sampling_rate=16000,
                return_tensors="pt", padding="max_length",
                return_attention_mask=True, truncation=True, 
                max_length= self.config.max_length,
            )
            inputs["labels"] = torch.tensor([0], dtype=torch.long)
            return inputs

        wavs, labels = zip(*valid_items)
        wavs = list(wavs)
        label_ids = [LABEL_TO_ID[lbl.strip().lower()] for lbl in labels]

        inputs = self.processor(
            wavs, sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            max_length= self.config.max_length,
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
        # LoRA 参数从 config 读取（支持 Ray Tune 调参）
        lora_r = getattr(config, "lora_r", 32)
        lora_alpha = getattr(config, "lora_alpha", 64)
        lora_dropout = getattr(config, "lora_dropout", 0.05)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="SEQ_CLS",
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        finetune_strategy = getattr(config, "finetune_strategy", "lora_only")
        if finetune_strategy == "classifier_only":
            # 使用模型自带的 emotion_classification_layer，不做替换
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
        # ---- 加载训练集 ----
        train_paths, train_labels = _concat_all_data_static(config.data_path_list)
        self.log("info", "[Data] Train samples: %s", len(train_paths))
        self.log("info", "[Data] Train label distribution: %s", dict(Counter(train_labels)))

        # ---- neutral 降采样（仅对 train） ----
        neutral_max = getattr(config, "neutral_max_samples", 0)
        if neutral_max > 0:
            neutral_idx = [i for i, l in enumerate(train_labels) if l == "neutral"]
            if len(neutral_idx) > neutral_max:
                rng = np.random.RandomState(config.seed)
                keep = set(rng.choice(neutral_idx, size=neutral_max, replace=False).tolist())
                train_paths = [p for i, p in enumerate(train_paths) if train_labels[i] != "neutral" or i in keep]
                train_labels = [l for i, l in enumerate(train_labels) if l != "neutral" or i in keep]
                self.log("info", "[Data] Neutral downsampled: %s → %s", len(neutral_idx), neutral_max)

        if config.resample_target_count and config.resample_labels:
            train_paths, train_labels = self._resample_minority(
                train_paths, train_labels,
                target_count=config.resample_target_count,
                minority_labels=config.resample_labels,
                seed=config.seed,
            )
            self.log("info", "[Data] After resample train labels: %s", dict(Counter(train_labels)))

        # ---- 加载评估集 ----
        eval_paths, eval_labels = _concat_all_data_static(config.eval_data_path_list)
        self.log("info", "[Data] Eval samples: %s", len(eval_paths))
        self.log("info", "[Data] Eval label distribution: %s", dict(Counter(eval_labels)))

        train_dataset = Dataset.from_dict({"path": train_paths, "label": train_labels})
        eval_dataset = Dataset.from_dict({"path": eval_paths, "label": eval_labels})
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
            # 显式传入全部 7 类标签，避免 sklearn 自动推断导致缺失类别
            from sklearn.metrics import precision_score
            all_possible_labels = list(range(NUM_EMOTIONS))
            per_class_recall = recall_score(all_labels, all_preds, labels=all_possible_labels, average=None, zero_division=0)
            per_class_precision = precision_score(all_labels, all_preds, labels=all_possible_labels, average=None, zero_division=0)
            per_class_f1 = f1_score(all_labels, all_preds, labels=all_possible_labels, average=None, zero_division=0)
            for i, emo in enumerate(EMO_MAP):
                emo_lower = emo.lower()
                metrics[f"Eval/{emo_lower}_precision"] = round(float(per_class_precision[i]), 4)
                metrics[f"Eval/{emo_lower}_recall"] = round(float(per_class_recall[i]), 4)
                metrics[f"Eval/{emo_lower}_f1"] = round(float(per_class_f1[i]), 4)
            # Debug: 打印混淆矩阵和标签分布
            from sklearn.metrics import confusion_matrix
            all_possible_labels = list(range(NUM_EMOTIONS))
            cm = confusion_matrix(all_labels, all_preds, labels=all_possible_labels)
            cm_lines = ["  Confusion Matrix (rows=true, cols=pred):"]
            cm_header = "           " + "".join(f"{e[:6]:>7}" for e in EMO_MAP)
            cm_lines.append(f"  {cm_header}")
            for i, emo in enumerate(EMO_MAP):
                row = "".join(f"{cm[i][j]:>7}" for j in range(NUM_EMOTIONS))
                cm_lines.append(f"  {emo.lower():<10}: {row}")
            logger.info(
                "Eval label distribution: %s\n%s",
                dict(Counter(all_labels)), "\n".join(cm_lines),
            )
        return metrics

    @staticmethod
    def load_model_for_inference(checkpoint_path: str, device: str = "cuda"):
        state_file = os.path.join(checkpoint_path, "training_state.pt")
        if not os.path.isfile(state_file):
            raise FileNotFoundError(f"Checkpoint not found: {state_file}")

        state = torch.load(state_file, map_location="cpu", weights_only=False)
        saved_config = state.get("config", {})
        finetune_strategy = saved_config.get("finetune_strategy", "lora_only")

        print(f"[Inference] Loading model from: {checkpoint_path}")
        print(f"[Inference] finetune_strategy={finetune_strategy}")

        repo = saved_config.get("model_name_or_path", "MERaLiON/MERaLiON-SER-v1")

        # 加载 processor
        processor = AutoProcessor.from_pretrained(
            repo, cache_dir=saved_config.get("cache_dir"),
            trust_remote_code=True, local_files_only=True,
        )
        # 加载基础模型
        model = AutoModelForAudioClassification.from_pretrained(
            repo, cache_dir=saved_config.get("cache_dir"),
            trust_remote_code=True, local_files_only=True,
            ignore_mismatched_sizes=True,
        )

        # 按训练时的策略注入 LoRA（如果需要）
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

        # 加载训练好的权重
        model.load_state_dict(state["model"], strict=False)
        model.to(device)
        model.eval()

        print(f"[Inference] Model loaded successfully, device={device}")
        return model, processor

    @staticmethod
    def evaluate_with_confusion_matrix(
        model, processor, data_path_list: List[str],
        audio_path_dir: str, device: str = "cuda",
        batch_size: int = 32,
        output_txt: Optional[str] = None,
    ) -> dict:
        from tqdm import tqdm as _tqdm
        from sklearn.metrics import (
            recall_score, f1_score, accuracy_score,
            precision_score, confusion_matrix,
        )

        model.eval()

        # 从外部文件加载数据
        all_paths, all_label_strs = _concat_all_data_static(data_path_list)
        label_ids = [LABEL_TO_ID[lbl.strip().lower()] for lbl in all_label_strs]
        n_total = len(all_paths)
        print(f"[EvalCM] Total samples: {n_total}, batch_size={batch_size}")

        all_preds: List[int] = []
        all_labels: List[int] = label_ids.copy()

        # ---- 批次推理 ----
        with torch.no_grad():
            for start in _tqdm(range(0, n_total, batch_size), desc="Inference (batch)"):
                end = min(start + batch_size, n_total)
                batch_paths = all_paths[start:end]

                # 批量加载 + 处理音频
                wavs = [_load_audio(p) for p in batch_paths]
                inputs = processor(
                    wavs, sampling_rate=16000,
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
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)

        # 计算指标
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_EMOTIONS)))
        acc = accuracy_score(all_labels, all_preds)
        uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        all_possible_labels = list(range(NUM_EMOTIONS))
        per_class_recall = recall_score(all_labels, all_preds, labels=all_possible_labels, average=None, zero_division=0)
        per_class_precision = precision_score(all_labels, all_preds, labels=all_possible_labels, average=None, zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, labels=all_possible_labels, average=None, zero_division=0)
        per_class_support = [int(sum(1 for l in all_labels if l == i)) for i in range(NUM_EMOTIONS)]

        results = {
            "accuracy": round(float(acc), 4),
            "uar": round(float(uar), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "confusion_matrix": cm,
            "per_class_precision": {EMO_MAP[i]: round(float(per_class_precision[i]), 4) for i in range(NUM_EMOTIONS)},
            "per_class_recall": {EMO_MAP[i]: round(float(per_class_recall[i]), 4) for i in range(NUM_EMOTIONS)},
            "per_class_f1": {EMO_MAP[i]: round(float(per_class_f1[i]), 4) for i in range(NUM_EMOTIONS)},
            "per_class_support": {EMO_MAP[i]: per_class_support[i] for i in range(NUM_EMOTIONS)},
            "preds": all_preds,
            "labels": all_labels,
        }

        # ---- 构建输出文本 ----
        lines: List[str] = []
        lines.append("=" * 65)
        lines.append("  FINE-TUNED MODEL EVALUATION")
        lines.append("=" * 65)
        lines.append(f"  Test samples  : {len(all_labels)}")
        lines.append(f"  Batch size    : {batch_size}")
        lines.append(f"  ── metrics ─────────────────────────────────────────────")
        lines.append(f"  ACC            : {acc*100:6.2f} %")
        lines.append(f"  UAR            : {uar*100:6.2f} %")
        lines.append(f"  F1 (macro)     : {f1_macro*100:6.2f} %")
        lines.append(f"  F1 (weighted)  : {f1_weighted*100:6.2f} %")
        lines.append("")
        lines.append(f"  Per-class:")
        lines.append(f"  {'class':<12} {'P':>7} {'R':>7} {'F1':>7} {'n':>6}")
        lines.append(f"  {'-'*42}")
        for i, emo in enumerate(EMO_MAP):
            p = per_class_precision[i] * 100
            r = per_class_recall[i] * 100
            f = per_class_f1[i] * 100
            n = per_class_support[i]
            lines.append(f"  {emo.lower():<12} {p:6.2f}% {r:6.2f}% {f:6.2f}% {n:>6}")
        lines.append(f"  {'-'*42}")
        lines.append("")
        lines.append("  Confusion Matrix (rows=true, cols=pred):")
        cm_header = "           " + "".join(f"{e[:6]:>7}" for e in EMO_MAP)
        lines.append(f"  {cm_header}")
        for i, emo in enumerate(EMO_MAP):
            row = "".join(f"{cm[i][j]:>7}" for j in range(NUM_EMOTIONS))
            lines.append(f"  {emo.lower():<10}: {row}")
        lines.append("=" * 65)
        lines.append("")

        # 逐样本预测详情
        lines.append("-" * 65)
        lines.append("  Per-sample predictions (path | true_label | pred_label | correct)")
        lines.append("-" * 65)
        correct_count = 0
        for i in range(n_total):
            true_lbl = ID_TO_LABEL[all_labels[i]]
            pred_lbl = ID_TO_LABEL[all_preds[i]]
            is_correct = "✓" if all_preds[i] == all_labels[i] else "✗"
            if all_preds[i] == all_labels[i]:
                correct_count += 1
            lines.append(f"  {all_paths[i]} | {true_lbl} | {pred_lbl} | {is_correct}")
        lines.append("-" * 65)
        lines.append(f"  Correct: {correct_count}/{n_total} ({100*correct_count/max(1,n_total):.2f}%)")
        lines.append("")

        # 输出：打印到终端 + 可选写入文件
        full_text = "\n".join(lines)
        print(full_text)

        if output_txt is not None:
            os.makedirs(os.path.dirname(output_txt) or ".", exist_ok=True)
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"[EvalCM] Results saved to: {output_txt}")

        return results

if __name__ == "__main__":
    import json
    # export HF_ENDPOINT=https://hf-mirror.com && CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 ddp_meralionser.py

    config = MeralionSERConfig()
    trial_params_json = os.environ.get("TRIAL_PARAMS", "")
    if trial_params_json:
        params = json.loads(trial_params_json)
        for k, v in params.items():
            if hasattr(config, k):
                setattr(config, k, v)
        print(f"[Trial] Overriding config: {json.dumps(params, indent=2)}")

    trainer = MeralionSERTrainer(config)
    trainer.train()