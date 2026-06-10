# CUDA 12.1 runtime + cuDNN8,匹配 torch 2.1+ cu121 轮子
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# 系统依赖:python + git(HF snapshot 用) + tini(PID1 信号转发) + 图像/OpenGL 运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip curl git tini \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# 先装依赖独立成层,代码变更时不重装包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷代码(.dockerignore 已排除 weights/data/tmp,镜像 ~ 8GB CUDA + ~5GB pip,不含权重)
COPY . .
RUN chmod +x /app/entrypoint.sh

EXPOSE 7890 8265

# 信号统一为 SIGTERM,由 ray_deploy._shutdown 监听并 drain inflight
STOPSIGNAL SIGTERM

# tini 做 PID1,负责信号转发和僵尸进程回收
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
