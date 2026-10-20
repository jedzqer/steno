"""冒烟测试：转录降级链、重试/媒体复用、下载跳过、任务数上限。

用法：python smoke_test.py
（不需要 pytest，只用项目自身的 fastapi/httpx 依赖；会写入并清理 videos/ 下的临时文件）
"""
import contextlib

from fastapi.testclient import TestClient
import webui  # 与本脚本同目录，运行时自动在 sys.path 上


def _cleanup():
    """清掉本脚本（或上一次运行）在 videos/ 下产生的临时文件。"""
    for p in list(webui.DOWNLOAD_DIR.glob("_smoke*")):
        with contextlib.suppress(OSError):
            p.unlink()


_cleanup()

# ---------- 1) 降级链的降级顺序 ----------
def prev(device="cuda:0", itn=True, chunk=120, retried=False):
    return webui._Attempt(device=device, use_itn=itn, chunk=chunk, overlap=1,
                          why="", retried=retried)

oom = RuntimeError("CUDA out of memory")
generic = RuntimeError("boom")

chain = []
a: webui._Attempt = prev()
for i in range(8):
    nxt = webui._next_attempt(a, generic, empty=(i == 1))
    if nxt is None:
        break
    chain.append((nxt["device"], nxt["use_itn"], nxt["chunk"], nxt["why"]))
    a = nxt
print("降级链(普通异常):")
for c in chain:
    print("   ", c)
# 链条是“失败后依次给出的下一次尝试”
assert chain[0][:3] == ("cuda:0", True, 120)          # 同参数重试
assert chain[0][3] == "同一参数重试一次"
assert chain[1][:3] == ("cuda:0", False, 120)          # 关 ITN
assert chain[1][3] == "关闭 ITN 后重试"
assert chain[2][:3] == ("cuda:0", False, 60)           # 分块减半
assert chain[2][3] == "分块减半后重试"
assert chain[3][:3] == ("cuda:0", False, 30)           # 再减半
assert chain[4][0] == "cpu"                            # 最后回落 CPU
assert chain[4][3] == "回落 CPU 重试"

# 空结果（无异常）时不浪费一次同参数重试，直接关 ITN
e = webui._next_attempt(prev(), None, empty=True)
assert e is not None and e["use_itn"] is False and e["chunk"] == 120, e
print("空结果降级:", e["use_itn"], e["chunk"], e["why"])

# OOM 直接回落 CPU
n = webui._next_attempt(prev(), oom, empty=False)
assert n is not None
assert n["device"] == "cpu" and "显存" in n["why"], n
print("OOM 降级:", n["device"], n["why"])

# ---------- 2) 用假 TV 驱动 _transcribe_with_fallbacks ----------
calls = []

class FakeTV:
    VIDEO_EXTENSIONS = webui.TV.VIDEO_EXTENSIONS
    AUDIO_EXTENSIONS = webui.TV.AUDIO_EXTENSIONS
    TranscriptionCancelled = webui.TV.TranscriptionCancelled
    def resolve_device(self, d): return "cpu" if d == "cpu" else "cuda:0"
    def load_model(self, device=None): return ("model", {})
    def transcribe_audio(self, path, model, kwargs, **kw):
        calls.append((kw.get("use_itn"), kw["chunk_sec"], kw.get("stop_event") is not None))
        if len(calls) == 1:
            raise oom
        if len(calls) == 2:
            return ""          # 空结果
        return "你好，这是转录结果"
webui.TV = FakeTV()
webui._MODELS.clear()
job = webui.Job("local", "x", {"device": "auto", "use_itn": True, "chunk": 120, "overlap": 1,
                               "language": "zh"})
text = webui._transcribe_with_fallbacks(job, "fake.wav")
print("降级链实际调用:", calls, "->", repr(text))
assert text == "你好，这是转录结果"
assert calls[0] == (True, 120, True)    # 首次：原参数
assert calls[1] == (True, 120, True)    # OOM → 回落 CPU（同参数）
assert calls[2] == (False, 120, True)   # 空结果 → 关 ITN

# ---------- 3) 重试接口：复用媒体 ----------
client = TestClient(webui.app)

media = webui.DOWNLOAD_DIR / "_smoke_media.m4a"
media.write_bytes(b"x" * 100)
j = webui.Job("url", "https://example.com/v", {"device": "auto", "use_itn": True,
                                               "chunk": 120, "overlap": 1, "language": "zh"})
j.status = "error"
j.media_path = str(media)
webui.JOBS[j.id] = j

r = client.post(f"/api/jobs/{j.id}/retry")
print("retry:", r.status_code, r.json())
assert r.status_code == 200 and r.json()["reused_media"] == "_smoke_media.m4a"
new_id = r.json()["id"]
assert webui.JOBS[new_id].kind == "local"
assert webui.JOBS[new_id].source == str(media)

# 媒体已删除、且不是 url 任务 -> 400
media.unlink()
j2 = webui.Job("local", str(media), {"device": "auto", "use_itn": True,
                                     "chunk": 120, "overlap": 1, "language": "zh"})
j2.status = "error"
j2.media_path = str(media)
webui.JOBS[j2.id] = j2
r = client.post(f"/api/jobs/{j2.id}/retry")
print("retry(本地媒体已删):", r.status_code, r.json())
assert r.status_code == 400

# url 任务且媒体已删 -> 重新下载（复用不了媒体）
r = client.post(f"/api/jobs/{j.id}/retry")
print("retry(url, 媒体已删):", r.status_code, r.json())
assert r.status_code == 200 and r.json()["reused_media"] is None
assert webui.JOBS[r.json()["id"]].kind == "url"

# ---------- 4) 历史重新转录 ----------
media2 = webui.DOWNLOAD_DIR / "_smoke_hist [ABC].m4a"
media2.write_bytes(b"y" * 50)
out = webui.DOWNLOAD_DIR / "_smoke_hist [ABC]_out.txt"
out.write_text("历史文本", encoding="utf-8")
r = client.post("/api/history/retry", params={"name": out.name})
print("history retry:", r.status_code, r.json())
assert r.status_code == 200 and r.json()["reused_media"] == media2.name
out.unlink()
media2.unlink()

# ---------- 5) JOBS 上限 ----------
for _ in range(250):
    jj = webui.Job("local", "x", {})
    jj.status = "done"
    webui.JOBS[jj.id] = jj
webui._prune_jobs()
print("JOBS after prune:", len(webui.JOBS))
assert len(webui.JOBS) <= webui.MAX_JOBS

# ---------- 6) 清理本脚本产生的临时文件 ----------
_cleanup()
print("\nALL PASSED (OK)")
