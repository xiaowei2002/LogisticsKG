"""基于内容哈希的磁盘 LLM 调用缓存，命中时跳过 LLM 调用以省去重复计费。"""
import hashlib
import json
import threading
from pathlib import Path

from loguru import logger


def cache_key(step: str, model: str, *parts: str) -> str:
    """按 step + model + 内容生成稳定哈希，作为缓存键。"""
    h = hashlib.sha256()
    h.update(step.encode("utf-8"))
    for p in (model,) + parts:
        h.update(b"\x00")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


class LLMCache:
    """基于内容哈希的磁盘缓存。

    键为 sha256(step | model | content...)，值为 JSON 可序列化结果。
    线程安全：并发写入时用锁串行化，避免字典在 json 序列化时被改动。
    """

    def __init__(self, path: Path | None):
        self.path = path
        self.hits = 0
        self.data: dict = {}
        self._lock = threading.Lock()
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("LLM 缓存文件损坏，已忽略: {}", path)
                self.data = {}

    def get(self, step: str, model: str, *parts: str):
        if self.path is None:
            return None
        key = cache_key(step, model, *parts)
        if key in self.data:
            self.hits += 1
            return self.data[key]
        return None

    def set(self, step: str, model: str, value, *parts: str) -> None:
        if self.path is None:
            return
        with self._lock:
            self.data[cache_key(step, model, *parts)] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.data, ensure_ascii=False), encoding="utf-8"
            )
