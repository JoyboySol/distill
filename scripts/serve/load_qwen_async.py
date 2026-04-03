#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
import string
import sys
import time
from collections import defaultdict, deque

import aiohttp


DEFAULT_TOPICS = [
    "请详细解释 Transformer 中 attention 的计算流程，并给出一个小例子。",
    "写一段关于自动驾驶安全验证流程的说明，要求分步骤展开。",
    "分析分布式训练中梯度同步与参数同步的区别，并说明适用场景。",
    "请从工程角度解释大模型推理服务的瓶颈，以及常见优化手段。",
    "写一篇关于数据管道监控体系建设的技术笔记，要求有小标题。",
]


def build_prompt(min_chars: int) -> str:
    topic = random.choice(DEFAULT_TOPICS)
    filler = []
    total = 0
    while total < min_chars:
        token = "".join(random.choices(string.ascii_lowercase + string.digits, k=64))
        filler.append(token)
        total += len(token) + 1

    return (
        f"{topic}\n\n"
        "请使用中文输出，内容尽量详细，包含定义、原理、示例、注意事项和总结。\n"
        "下面这些随机片段只是为了拉长上下文，请忽略它们本身的语义，但要继续完成任务：\n"
        + "\n".join(filler)
    )


class Stats:
    def __init__(self, latency_window_size: int = 5000) -> None:
        self.start_time = time.time()
        self.total = 0
        self.success = 0
        self.failed = 0
        self.inflight = 0

        self.last_error = ""
        self.last_port = None
        self.last_latency = 0.0

        self.port_success = defaultdict(int)
        self.port_failed = defaultdict(int)

        self.latencies = deque(maxlen=latency_window_size)
        self.lock = asyncio.Lock()

    async def on_start(self) -> None:
        async with self.lock:
            self.inflight += 1

    async def on_done(self, ok: bool, port: int, latency: float, err_msg: str = "") -> None:
        async with self.lock:
            self.inflight -= 1
            self.total += 1
            self.last_port = port
            self.last_latency = latency
            self.latencies.append(latency)

            if ok:
                self.success += 1
                self.port_success[port] += 1
            else:
                self.failed += 1
                self.last_error = err_msg
                self.port_failed[port] += 1

    async def snapshot(self) -> dict:
        async with self.lock:
            elapsed = max(time.time() - self.start_time, 1e-9)
            total = self.total
            success = self.success
            failed = self.failed
            inflight = self.inflight
            last_port = self.last_port
            last_latency = self.last_latency
            last_error = self.last_error
            latencies = list(self.latencies)
            port_success = dict(self.port_success)
            port_failed = dict(self.port_failed)

        lat_sorted = sorted(latencies)
        avg_latency = sum(lat_sorted) / len(lat_sorted) if lat_sorted else 0.0
        p50 = percentile(lat_sorted, 50)
        p95 = percentile(lat_sorted, 95)
        p99 = percentile(lat_sorted, 99)

        return {
            "elapsed": elapsed,
            "total": total,
            "success": success,
            "failed": failed,
            "inflight": inflight,
            "rps": total / elapsed,
            "avg_latency": avg_latency,
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            "last_port": last_port,
            "last_latency": last_latency,
            "last_error": last_error,
            "port_success": port_success,
            "port_failed": port_failed,
        }


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int(round((p / 100.0) * (len(values) - 1)))
    return values[idx]


def make_payload(args) -> dict:
    prompt = build_prompt(args.prompt_chars)
    return {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": "你是一个严谨的中文助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": False,
    }


async def post_json(session: aiohttp.ClientSession, url: str, payload: dict, timeout_sec: int) -> dict:
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"http {resp.status} body={text[:300]}")
        return json.loads(text)


async def one_request(
    req_id: int,
    args,
    stats: Stats,
    session: aiohttp.ClientSession,
) -> None:
    port = args.ports[req_id % len(args.ports)]
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = make_payload(args)

    await stats.on_start()
    start = time.time()
    ok = False
    err_msg = ""

    try:
        data = await post_json(session, url, payload, args.timeout)
        ok = "choices" in data
        if not ok:
            err_msg = f"bad response keys={list(data.keys())[:5]}"
    except Exception as exc:  # noqa: BLE001
        err_msg = repr(exc)

    latency = time.time() - start
    await stats.on_done(ok=ok, port=port, latency=latency, err_msg=err_msg)

    if (not ok) and args.sleep_on_error > 0:
        await asyncio.sleep(args.sleep_on_error)


async def request_producer(args, stats: Stats) -> None:
    connector = aiohttp.TCPConnector(
        limit=0,
        limit_per_host=0,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        force_close=False,
    )

    headers = {"Content-Type": "application/json"}
    req_id = 0
    pending: set[asyncio.Task] = set()

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        while True:
            if args.duration > 0 and (time.time() - stats.start_time) >= args.duration:
                break

            while len(pending) < args.concurrency:
                task = asyncio.create_task(one_request(req_id, args, stats, session))
                pending.add(task)
                task.add_done_callback(pending.discard)
                req_id += 1

            if pending:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0.01)

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def reporter(args, stats: Stats) -> None:
    while True:
        await asyncio.sleep(args.report_interval)
        snap = await stats.snapshot()

        elapsed = int(snap["elapsed"])
        print(
            f"[{elapsed}s] "
            f"total={snap['total']} success={snap['success']} failed={snap['failed']} "
            f"inflight={snap['inflight']} rps={snap['rps']:.2f} "
            f"avg={snap['avg_latency']:.2f}s p50={snap['p50_latency']:.2f}s "
            f"p95={snap['p95_latency']:.2f}s p99={snap['p99_latency']:.2f}s "
            f"last_port={snap['last_port']} last_latency={snap['last_latency']:.2f}s "
            f"last_error={snap['last_error']}",
            flush=True,
        )

        if args.duration > 0 and snap["elapsed"] >= args.duration and snap["inflight"] == 0:
            break


async def main_async(args) -> int:
    stats = Stats(latency_window_size=args.latency_window)

    print(
        f"Starting async load: ports={args.ports} model={args.model_name} "
        f"concurrency={args.concurrency} prompt_chars={args.prompt_chars} "
        f"max_tokens={args.max_tokens} duration={args.duration}",
        flush=True,
    )

    if args.warmup_seconds > 0:
        print(f"Warming up for {args.warmup_seconds}s...", flush=True)
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.duration = args.warmup_seconds
        warmup_stats = Stats(latency_window_size=min(1000, args.latency_window))
        await asyncio.gather(
            request_producer(warmup_args, warmup_stats),
            reporter(warmup_args, warmup_stats),
        )
        print("Warmup done. Starting measured run...", flush=True)

    await asyncio.gather(
        request_producer(args, stats),
        reporter(args, stats),
    )

    snap = await stats.snapshot()
    print("\nFinal summary:", flush=True)
    print(json.dumps(snap, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="High-concurrency async load generator for local vLLM/Qwen servers.")
    parser.add_argument("--ports", nargs="+", type=int, default=[6758, 6759, 6760, 6761])
    parser.add_argument("--model-name", default="Qwen3-14B")

    parser.add_argument("--concurrency", type=int, default=128, help="Number of in-flight requests.")
    parser.add_argument("--duration", type=int, default=0, help="Run seconds, 0 means forever.")

    parser.add_argument("--prompt-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=1800)

    parser.add_argument("--sleep-on-error", type=float, default=0.2)
    parser.add_argument("--report-interval", type=int, default=5)
    parser.add_argument("--latency-window", type=int, default=5000)
    parser.add_argument("--warmup-seconds", type=int, default=0)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Stopping load generator...", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())