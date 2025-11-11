import os
from typing import Literal, Optional, Iterator, Any, TypedDict, List, Union
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
import logging
from contextlib import contextmanager


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class CoreLLM:
    """
    核心 LLM 封装，支持流式与非流式调用，统一错误处理与日志
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        **default_kwargs: Any,
    ):
        """
        初始化 CoreLLM

        Args:
            model_name: 模型名称
            api_key: API 密钥
            base_url: 可选，自定义服务地址（如本地部署）
            logger: 日志记录器
            **default_kwargs: 默认调用参数（如 max_tokens, temperature, stream 等）
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)

        # 存储默认参数（仅保留 OpenAI 支持的参数）
        self.default_kwargs = {
            k: v for k, v in default_kwargs.items()
            if k in {
                "max_tokens", "temperature", "top_p", "stream",
                "timeout", "extra_kwargs"
            } and v is not None
        }
        # 确保 extra_kwargs 是 dict
        if "extra_kwargs" not in self.default_kwargs:
            self.default_kwargs["extra_kwargs"] = {}

        # 延迟初始化 client
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=30.0,  # 默认超时
            )
        return self._client

    def _build_call_kwargs(self, **override_kwargs) -> dict:
        """合并默认参数与本次调用参数"""
        # 浅拷贝默认参数
        call_kwargs = self.default_kwargs.copy()

        # 合并 extra_kwargs
        extra = call_kwargs.pop("extra_kwargs", {}).copy()
        if "extra_kwargs" in override_kwargs:
            extra.update(override_kwargs.pop("extra_kwargs") or {})

        # 合并其他参数（override_kwargs 优先）
        call_kwargs.update({
            k: v for k, v in override_kwargs.items()
            if v is not None and k != "extra_kwargs"
        })

        # 放回 extra_kwargs
        if extra:
            call_kwargs["extra_kwargs"] = extra

        return call_kwargs

    def llm_out(
        self,
        messages: List[Message],
        **kwargs,
    ) -> Union[str, Iterator[str]]:
        """
        调用 LLM

        Args:
            messages: 消息列表
            **kwargs: 覆盖默认参数（如 max_tokens, stream 等）

        Returns:
            str: 非流式时返回完整内容
            Iterator[str]: 流式时返回内容块迭代器
        """
        call_kwargs = self._build_call_kwargs(**kwargs)
        stream = call_kwargs.get("stream", False)
        extra_kwargs = call_kwargs.pop("extra_kwargs", {})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
                **call_kwargs,
                **extra_kwargs,
            )

            if stream:
                def stream_generator() -> Iterator[str]:
                    collected = ""
                    for chunk in response:
                        content = (chunk.choices[0].delta.content or "")
                        if content:
                            collected += content
                            yield content
                    self.logger.debug(f"Stream completed: {len(collected)} chars")

                return stream_generator()

            else:
                content = response.choices[0].message.content or ""
                preview = content[:200] + ("..." if len(content) > 200 else "")
                self.logger.debug(f"LLM response: {preview}")
                return content

        except AuthenticationError:
            self.logger.error("LLM 认证失败，请检查 API Key")
            raise
        except RateLimitError:
            self.logger.warning("触发频率限制，建议稍后重试或升级配额")
            raise
        except APIError as e:
            self.logger.error(f"LLM API 错误: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"LLM 调用未知异常: {type(e).__name__}: {e}")
            raise

    @contextmanager
    def session(self):
        """支持 with 语句（可扩展资源管理）"""
        try:
            yield self
        finally:
            pass


if __name__ == '__main__':
    # 示例：豆包模型（火山引擎）
    llm_core = CoreLLM(
        model_name='doubao-seed-1-6-flash-250615',
        api_key='636b7b0b-2fe5-477d-9caf-ede47e5ab26e',
        base_url='https://ark.cn-beijing.volces.com/api/v3',
        max_tokens=1024,
        temperature=0.7,
    )

    out = llm_core.llm_out([
        {"role": "user", "content": "你好，世界！"}
    ])
    print(out)