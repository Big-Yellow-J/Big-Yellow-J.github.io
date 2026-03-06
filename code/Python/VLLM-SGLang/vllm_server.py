import os
import uvicorn
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from vllm import LLM, SamplingParams

def load_model(model_name: str, cache_dir: str = "/hy-tmp") -> LLM:
    """
    初始化 vLLM 模型，适合常驻服务。
    """
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        download_dir=cache_dir,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_num_seqs=16,
        max_model_len=1024,
        logprobs_mode="processed_logprobs",
        enable_sleep_mode=True,  # 空闲自动休眠
        distributed_executor_backend=None,
    )
    return llm

MODEL_NAME = "Qwen/Qwen3-0.6B"
llm_model = load_model(MODEL_NAME)
app = FastAPI(title="vLLM API Server")


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="输入的文本")
    max_tokens: int = Field(128, ge=1, le=2048, description="生成的最大 token 数")
    n: int = Field(1, ge=1, le=16, description="生成的候选序列数")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p 采样概率")

class GenerationResponse(BaseModel):
    results: List[str]

@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    try:
        sampling_params = SamplingParams(
            max_tokens=req.max_tokens,
            n=req.n,
            temperature=req.temperature,
            top_p=req.top_p
        )
        outputs = llm_model.generate([req.prompt], sampling_params)
        results = [o.outputs[0].text for o in outputs]
        return GenerationResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("vllm_server:app", host="0.0.0.0", port=8000, log_level="info")
