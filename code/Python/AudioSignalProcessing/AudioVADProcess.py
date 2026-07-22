from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


def _mask_to_segments(mask: np.ndarray, sr: int, hop_len: int,
                      min_speech_ms: int, min_silence_ms: int):
    min_speech_frames = max(1, int(min_speech_ms * sr / 1000 / hop_len))
    min_silence_frames = max(1, int(min_silence_ms * sr / 1000 / hop_len))
    changes = np.diff(np.pad(mask.astype(int), 1))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    segments = []
    for s, e in zip(starts, ends):
        if e - s < min_speech_frames:
            continue
        if segments and s - segments[-1][1] < min_silence_frames:
            segments[-1] = (segments[-1][0], e)
        else:
            segments.append((s, e))

    hop_s = hop_len / sr
    return [VADSegment(s * hop_s, e * hop_s) for s, e in segments]


@dataclass
class VADSegment:
    start: float
    end: float
    label: str = "speech"


class BaseVAD(ABC):
    @abstractmethod
    def detect(self, audio: Union[str, np.ndarray], sr: int = 16000) -> List[VADSegment]: ...


class EnergyVAD(BaseVAD):
    def __init__(self, threshold_db: float = -30, frame_ms: int = 25, hop_ms: int = 10,
                 min_speech_ms: int = 200, min_silence_ms: int = 300):
        self.threshold_db = threshold_db
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

    def detect(self, audio: Union[str, np.ndarray], sr: int = 16000) -> List[VADSegment]:
        import librosa
        y = librosa.load(audio, sr=sr, mono=True)[0] if isinstance(audio, (str, Path)) else audio
        frame_len = int(self.frame_ms * sr / 1000)
        hop_len = int(self.hop_ms * sr / 1000)
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        db = librosa.amplitude_to_db(rms, ref=np.max)
        return _mask_to_segments(db > self.threshold_db, sr, hop_len, self.min_speech_ms, self.min_silence_ms)


class WebRTCVAD(BaseVAD):
    def __init__(self, aggressiveness: int = 2):
        from webrtcvad import Vad
        self.vad = Vad(aggressiveness)

    def detect(self, audio: Union[str, np.ndarray], sr: int = 16000) -> List[VADSegment]:
        import librosa
        y = librosa.load(audio, sr=sr, mono=True)[0] if isinstance(audio, (str, Path)) else audio
        y = (y * 32767).astype(np.int16)
        frame_ms = 30
        data = y.tobytes() if isinstance(y, np.ndarray) else np.array(y).tobytes()
        n = int(sr * frame_ms / 1000) * 2
        speech_mask = [self.vad.is_speech(data[i:i + n], sr) for i in range(0, len(data) - n + 1, n)]
        return _mask_to_segments(np.array(speech_mask), sr, n // 2, 200, 300)


class SileroVAD(BaseVAD):
    def __init__(self, use_onnx: bool = False, threshold: float = 0.5,
                 min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        try:
            from silero_vad import load_silero_vad
            self.model = load_silero_vad(onnx=use_onnx)
            self._use_lib = True
        except ImportError:
            import torch
            model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
            self.model = model
            self._utils = utils
            self._use_lib = False

    def detect(self, audio: Union[str, np.ndarray], sr: int = 16000) -> List[VADSegment]:
        kwargs = dict(threshold=self.threshold, min_speech_duration_ms=self.min_speech_duration_ms,
                      min_silence_duration_ms=self.min_silence_duration_ms,
                      speech_pad_ms=self.speech_pad_ms, return_seconds=True)
        if self._use_lib:
            from silero_vad import get_speech_timestamps, read_audio
            wav = read_audio(audio, sr) if isinstance(audio, (str, Path)) else audio
            return [VADSegment(t["start"], t["end"]) for t in get_speech_timestamps(wav, self.model, **kwargs)]

        get_speech_ts, _, read_audio, _, _ = self._utils
        wav = read_audio(audio, sr) if isinstance(audio, (str, Path)) else audio
        return [VADSegment(t["start"], t["end"]) for t in get_speech_ts(wav, self.model, **kwargs)]


class FSMNVAD(BaseVAD):
    def __init__(self, max_single_segment_time: int = 15000, device: str = "cpu",
                 sample_rate: int = 16000, frame_in_ms: int = 10, frame_out_ms: int = 25,
                 window_size_ms: int = 200, detect_mode: int = 0,
                 do_start_point_detection: bool = True, do_end_point_detection: bool = True,
                 max_start_silence_time: int = 0, max_end_silence_time: int = 0,
                 lookback_time_start_point: int = 0, lookahead_time_end_point: int = 0,
                 sil_to_speech_time_thres: int = 150, speech_to_sil_time_thres: int = 150,
                 do_extend: int = 1, snr_mode: int = 0, snr_thres: float = -100,
                 noise_frame_num_used_for_snr: int = 100, decibel_thres: int = -100,
                 speech_noise_thres: float = 0.8,
                 speech_noise_thresh_low: float = -1, speech_noise_thresh_high: float = 1,
                 speech_2_noise_ratio: float = 0.5, fe_prior_thres: float = 1e-4):
        from funasr import AutoModel
        self.model = AutoModel(
            model="fsmn-vad", device=device,
            sample_rate=sample_rate, frame_in_ms=frame_in_ms, frame_out_ms=frame_out_ms,
            window_size_ms=window_size_ms, detect_mode=detect_mode,
            do_start_point_detection=do_start_point_detection,
            do_end_point_detection=do_end_point_detection,
            max_start_silence_time=max_start_silence_time,
            max_end_silence_time=max_end_silence_time,
            lookback_time_start_point=lookback_time_start_point,
            lookahead_time_end_point=lookahead_time_end_point,
            sil_to_speech_time_thres=sil_to_speech_time_thres,
            speech_to_sil_time_thres=speech_to_sil_time_thres,
            do_extend=do_extend, max_single_segment_time=max_single_segment_time,
            snr_mode=snr_mode, snr_thres=snr_thres,
            noise_frame_num_used_for_snr=noise_frame_num_used_for_snr,
            decibel_thres=decibel_thres, speech_noise_thres=speech_noise_thres,
            speech_noise_thresh_low=speech_noise_thresh_low,
            speech_noise_thresh_high=speech_noise_thresh_high,
            speech_2_noise_ratio=speech_2_noise_ratio, fe_prior_thres=fe_prior_thres,
        )

    def detect(self, audio: Union[str, np.ndarray], sr: int = 16000) -> List[VADSegment]:
        result = self.model.generate(input=audio if isinstance(audio, (str, Path)) else audio)
        if not result or "value" not in result[0]:
            return []
        return [VADSegment(seg[0] / 1000, seg[1] / 1000) for seg in result[0]["value"]]


class PyannoteVAD(BaseVAD):
    def __init__(self, token: Optional[str] = None):
        import os
        token = token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("请提供 HF token 或设置环境变量 HF_TOKEN")
        from huggingface_hub import login
        login(token=token)
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

    def detect(self, audio: Union[str, np.ndarray], sr: int = 16000) -> List[VADSegment]:
        result = self.pipeline(audio if isinstance(audio, (str, Path)) else {"waveform": audio, "sample_rate": sr})
        return [VADSegment(seg.start, seg.end) for seg in result.itersegments()]


class FireRedVAD(BaseVAD):
    _DEFAULT_DIR = "pretrained_models/FireRedVAD/VAD"

    def __init__(self, model_dir: Optional[str] = None,
                 speech_threshold: float = 0.4, use_gpu: bool = False,
                 smooth_window_size: int = 5, min_speech_frame: int = 20,
                 max_speech_frame: int = 2000, min_silence_frame: int = 20,
                 merge_silence_frame: int = 0, extend_speech_frame: int = 0,
                 chunk_max_frame: int = 30000):
        model_dir = model_dir or self._DEFAULT_DIR
        if not Path(model_dir, "cmvn.ark").exists():
            self._download(model_dir)
        from fireredvad import FireRedVad, FireRedVadConfig

        config = FireRedVadConfig(
            use_gpu=use_gpu, speech_threshold=speech_threshold,
            smooth_window_size=smooth_window_size,
            min_speech_frame=min_speech_frame, max_speech_frame=max_speech_frame,
            min_silence_frame=min_silence_frame, merge_silence_frame=merge_silence_frame,
            extend_speech_frame=extend_speech_frame, chunk_max_frame=chunk_max_frame,
        )
        self.vad = FireRedVad.from_pretrained(model_dir, config)

    @staticmethod
    def _download(target_dir: str):
        from modelscope import snapshot_download
        parent = str(Path(target_dir).parent)
        print(f"正在下载 FireRedVAD 模型到 {parent} ...")
        snapshot_download("xukaituo/FireRedVAD", local_dir=parent)
        if not Path(target_dir, "cmvn.ark").exists():
            raise FileNotFoundError(f"模型下载失败, 请手动: modelscope download --model xukaituo/FireRedVAD --local_dir {parent}")

    def detect(self, audio: Union[str, np.ndarray], sr: int = 16000) -> List[VADSegment]:
        if isinstance(audio, np.ndarray):
            import tempfile
            import soundfile as sf
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio, sr)
            audio = tmp.name
        result, _ = self.vad.detect(audio)
        return [VADSegment(float(s), float(e)) for s, e in result["timestamps"]]


class AuditokVAD(BaseVAD):
    def __init__(self, energy_threshold: float = 50, min_dur: float = 0.2,
                 max_dur: float = 15, max_silence: float = 0.3):
        self.energy_threshold = energy_threshold
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.max_silence = max_silence

    def detect(self, audio: Union[str, np.ndarray], sr: int = 16000) -> List[VADSegment]:
        import tempfile
        import soundfile as sf
        import auditok
        if isinstance(audio, (str, Path)):
            source = str(audio)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio, sr)
            source = tmp.name
        regions = auditok.split(
            source, energy_threshold=self.energy_threshold,
            min_dur=self.min_dur, max_dur=self.max_dur, max_silence=self.max_silence)
        return [VADSegment(r.start, r.end) for r in regions]


_REGISTRY = {
    "energy": EnergyVAD, "webrtc": WebRTCVAD, "silero": SileroVAD,
    "fsmn": FSMNVAD, "pyannote": PyannoteVAD, "firered": FireRedVAD, "auditok": AuditokVAD,
}


class VADProcess:
    @staticmethod
    def create(name: str = "silero", **kwargs) -> BaseVAD:
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"未知 VAD: {name}, 可选: {list(_REGISTRY)}")
        return cls(**kwargs)

    @staticmethod
    def list_models() -> List[str]:
        return list(_REGISTRY.keys())

    @staticmethod
    def process_all(audio: Union[str, np.ndarray], sr: int = 16000,
                    skip: Optional[List[str]] = None) -> dict:
        """依次运行所有可用 VAD, 返回 {模型名: [VADSegment, ...]}"""
        skip = skip or []
        results = {}
        for name in _REGISTRY:
            if name in skip:
                continue
            try:
                vad = VADProcess.create(name)
                results[name] = vad.detect(audio, sr)
            except Exception as e:
                results[name] = f"加载失败: {e}"
        return results

    @staticmethod
    def store(audio: Union[str, np.ndarray], output_dir: str = "./vad_output",
              sr: int = 16000, skip: Optional[List[str]] = None):
        """运行所有 VAD, 按模型切分音频并保存为 wav"""
        import os
        import soundfile as sf

        y = VADProcess._load_audio(audio, sr)
        audio_path = audio if isinstance(audio, (str, Path)) else "audio"
        audio_name = Path(audio_path).stem if isinstance(audio_path, (str, Path)) else "audio"
        base = os.path.join(output_dir, audio_name)
        os.makedirs(base, exist_ok=True)

        results = VADProcess.process_all(y, sr, skip)
        for model_name, segs in results.items():
            if isinstance(segs, str):
                print(f"[{model_name}] {segs}")
                continue
            model_dir = os.path.join(base, model_name)
            os.makedirs(model_dir, exist_ok=True)
            for i, seg in enumerate(segs):
                try:
                    chunk = y[int(float(seg.start) * sr):int(float(seg.end) * sr)]
                    out = os.path.join(model_dir, f"{i:03d}_{float(seg.start):.2f}s-{float(seg.end):.2f}s.wav")
                    sf.write(out, chunk, sr)
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"  ⚠ [{model_name}] 第 {i} 段数据异常: {seg} -> {e}")
            print(f"[{model_name}] {len(segs)} 段 → {model_dir}")

            VADProcess.plot(audio_path, y, sr, segs, model_name, model_dir)

        print(f"\n已保存至: {base}")

    @staticmethod
    def plot(audio_path: str, y: np.ndarray, sr: int,
             segments: List[VADSegment], model_name: str, save_dir: str):
        """绘制 Mel 频谱图并叠加 VAD 切分矩形框"""
        import os
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if not segments:
            return
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(14, 5))
        librosa.display.specshow(mel_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel", cmap="magma")
        plt.colorbar(label="dB")

        for i, seg in enumerate(segments):
            plt.gca().add_patch(Rectangle(
                (seg.start, 0), seg.end - seg.start, sr // 2,
                linewidth=2, edgecolor="red", facecolor="none"))

        plt.title(f"{Path(audio_path).stem} — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_name}_mel.png"), dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _load_audio(audio: Union[str, np.ndarray], sr: int) -> np.ndarray:
        import librosa
        return librosa.load(audio, sr=sr, mono=True)[0] if isinstance(audio, (str, Path)) else audio


if __name__ == "__main__":
    VADProcess.store("./audio.wav", "./vad_output")