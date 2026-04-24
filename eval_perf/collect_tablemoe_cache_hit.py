import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR_STR = str(SCRIPT_DIR)
if SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR_STR)

from common import (
    MODEL_SPECS,
    DATASET_LOADERS,
    PROMPT_BUILDERS,
    _build_runtime,
    _cleanup_runtime,
    _resolve_sample_id,
    _resolve_source_index,
    _set_cache_env,
    detect_dataset_key,
    sample_dataset_for_benchmark,
)
from moe_infinity_common import _prepare_deepseek_inputs, _prepare_qwen_inputs, resolve_image_root
from table_moe.utils import ModalityContext, TimingStreamer


METHOD_ORDER = ["adapmoe", "skip", "offline", "online", "tablemoe"]
METHOD_LABELS = {
    "adapmoe": "AdapMoE",
    "skip": "AdapMoE(+gating)",
    "offline": "+ALUT",
    "online": "+WINDOW",
    "tablemoe": "TableMoE",
}
METHOD_ALIASES = {
    "adapmoe": "adapmoe",
    "baseline": "adapmoe",
    "skip": "skip",
    "skipoffload": "skip",
    "skip_offload": "skip",
    "offline": "offline",
    "online": "online",
    "tablemoe": "tablemoe",
    "hybrid": "tablemoe",
}

CACHE_HIT_DECODE_PREFETCH_LIMIT = 2


class LinearCacheStopWatch(TimingStreamer):
    def __init__(self, expert_cache):
        super().__init__()
        self.expert_cache = expert_cache
        self.decode_started = False

    def _on_decode_start(self):
        self.decode_started = True
        if self.expert_cache is not None and hasattr(self.expert_cache, "clear_cache_stats"):
            self.expert_cache.clear_cache_stats()

    def get_decode_cache_stats(self):
        empty = {
            "hits": 0,
            "resident_hits": 0,
            "prefetched_hits": 0,
            "misses": 0,
            "total": 0,
            "hit_rate": 0.0,
        }
        if not self.decode_started:
            return empty
        if self.expert_cache is None or not hasattr(self.expert_cache, "get_cache_stats"):
            return empty
        return self.expert_cache.get_cache_stats()


@contextmanager
def cache_hit_decode_prefetch_override(model_key: str):
    if model_key == "qwen3vlmoe":
        from table_moe.models.qwen3_vl_moe import get_offload_config, update_offload_config

        get_config = get_offload_config
        update_config = update_offload_config
    else:
        from table_moe.models.deepseek_vl2 import (
            get_deepseek_offload_config,
            update_deepseek_offload_config,
        )

        get_config = get_deepseek_offload_config
        update_config = update_deepseek_offload_config

    previous_config = get_config()
    try:
        # Keep this override private to cache-hit collection so normal eval paths stay unchanged.
        update_config(
            prefetch=True,
            prefetch_limit_decode=CACHE_HIT_DECODE_PREFETCH_LIMIT,
        )
        yield
    finally:
        update_config(**previous_config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect decode-only cache-hit statistics for AdapMoE/skip/offline/online/TableMoE."
    )
    parser.add_argument("--model", type=str, required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--pca-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--methods", "--modes", dest="methods", type=str, default=",".join(METHOD_ORDER))
    parser.add_argument("--sample-ratio", type=float, default=0.01)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=7)
    parser.add_argument("--cache-ratio", type=float, default=0.5)
    parser.add_argument("--keep-rate", type=float, default=0.6)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument(
        "--deepseek-repo",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "third_party" / "DeepSeek-VL2"),
    )
    return parser.parse_args()


def normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum() or ch == "_")


def resolve_methods(raw: str) -> list[str]:
    methods = []
    for item in str(raw).replace(",", " ").split():
        key = METHOD_ALIASES.get(normalize_token(item))
        if key is None:
            raise ValueError(f"Unsupported cache-hit method: {item}")
        if key not in methods:
            methods.append(key)
    if not methods:
        raise ValueError("At least one cache-hit method is required")
    return [method for method in METHOD_ORDER if method in methods]


def maybe_sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_dataset(data_dir: str):
    dataset_key = detect_dataset_key(data_dir)
    dataset = DATASET_LOADERS[dataset_key](data_dir)
    if "__source_index__" not in dataset.columns:
        dataset = dataset.copy()
        dataset["__source_index__"] = dataset.index
    benchmark = Path(data_dir).stem
    return dataset_key, benchmark, dataset


def build_sample_metric(sample_id, source_index, ttft, tpot, num_new_tokens, cache_stats):
    return {
        "id": str(sample_id),
        "source_index": str(source_index),
        "ttft": float(ttft),
        "tpot": float(tpot),
        "num_new_tokens": int(num_new_tokens),
        "cache_hits": int(cache_stats.get("hits", 0)),
        "cache_resident_hits": int(cache_stats.get("resident_hits", 0)),
        "cache_prefetched_hits": int(cache_stats.get("prefetched_hits", 0)),
        "cache_misses": int(cache_stats.get("misses", 0)),
        "cache_total": int(cache_stats.get("total", 0)),
        "cache_hit_rate": float(cache_stats.get("hit_rate", 0.0)),
    }


def init_aggregate(label: str):
    return {
        "label": label,
        "hits": 0,
        "resident_hits": 0,
        "prefetched_hits": 0,
        "misses": 0,
        "total": 0,
        "num_samples": 0,
        "avg_num_new_tokens": 0.0,
        "avg_ttft": 0.0,
        "avg_tpot": 0.0,
    }


def update_aggregate(aggregate: dict, metric: dict):
    aggregate["hits"] += int(metric["cache_hits"])
    aggregate["resident_hits"] += int(metric["cache_resident_hits"])
    aggregate["prefetched_hits"] += int(metric["cache_prefetched_hits"])
    aggregate["misses"] += int(metric["cache_misses"])
    aggregate["total"] += int(metric["cache_total"])
    aggregate["num_samples"] += 1
    aggregate["avg_num_new_tokens"] += float(metric["num_new_tokens"])
    aggregate["avg_ttft"] += float(metric["ttft"])
    aggregate["avg_tpot"] += float(metric["tpot"])


def finalize_aggregate(aggregate: dict):
    num_samples = max(1, int(aggregate["num_samples"]))
    aggregate["avg_num_new_tokens"] /= num_samples
    aggregate["avg_ttft"] /= num_samples
    aggregate["avg_tpot"] /= num_samples
    total = int(aggregate["total"])
    hits = int(aggregate["hits"])
    misses = int(aggregate["misses"])
    aggregate["hit_rate"] = float(hits) / float(total) if total > 0 else 0.0
    aggregate["miss_rate"] = float(misses) / float(total) if total > 0 else 0.0
    return aggregate


def run_qwen_warmup(runtime, dataset, build_prompt, img_root, warmup_samples):
    model = runtime["model"]
    num_warmup = min(int(warmup_samples), len(dataset))
    for idx in range(num_warmup):
        row = dataset.iloc[idx]
        inputs = _prepare_qwen_inputs(runtime, row, build_prompt, img_root, padding=True)
        ModalityContext.set_input_ids(inputs.input_ids)
        try:
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        finally:
            ModalityContext.clear()


def run_deepseek_warmup(runtime, dataset, build_prompt, img_root, warmup_samples):
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    num_warmup = min(int(warmup_samples), len(dataset))
    for idx in range(num_warmup):
        row = dataset.iloc[idx]
        prepare_inputs = _prepare_deepseek_inputs(runtime, row, build_prompt, img_root)
        ModalityContext.set_input_ids(getattr(prepare_inputs, "input_ids", None))
        try:
            with torch.no_grad():
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                _ = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=8,
                    do_sample=False,
                    use_cache=True,
                )
        finally:
            ModalityContext.clear()


def run_qwen_sample(runtime, row, build_prompt, img_root, args, sample_id, source_index):
    model = runtime["model"]
    inputs = _prepare_qwen_inputs(runtime, row, build_prompt, img_root, padding=False)
    expert_cache = runtime.get("expert_cache")
    streamer = LinearCacheStopWatch(expert_cache)

    if expert_cache is not None and hasattr(expert_cache, "clear_cache_stats"):
        expert_cache.clear_cache_stats()

    maybe_sync_cuda()
    ModalityContext.set_input_ids(inputs.input_ids)
    try:
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )
    finally:
        ModalityContext.clear()
    maybe_sync_cuda()

    ttft, tpot, num_new_tokens = streamer.get_metrics()
    return build_sample_metric(
        sample_id=sample_id,
        source_index=source_index,
        ttft=ttft,
        tpot=tpot,
        num_new_tokens=num_new_tokens,
        cache_stats=streamer.get_decode_cache_stats(),
    )


def run_deepseek_sample(runtime, row, build_prompt, img_root, args, sample_id, source_index):
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    prepare_inputs = _prepare_deepseek_inputs(runtime, row, build_prompt, img_root)
    expert_cache = runtime.get("expert_cache")
    streamer = LinearCacheStopWatch(expert_cache)

    if expert_cache is not None and hasattr(expert_cache, "clear_cache_stats"):
        expert_cache.clear_cache_stats()

    maybe_sync_cuda()
    ModalityContext.set_input_ids(getattr(prepare_inputs, "input_ids", None))
    try:
        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            _ = model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                streamer=streamer,
            )
    finally:
        ModalityContext.clear()
    maybe_sync_cuda()

    ttft, tpot, num_new_tokens = streamer.get_metrics()
    return build_sample_metric(
        sample_id=sample_id,
        source_index=source_index,
        ttft=ttft,
        tpot=tpot,
        num_new_tokens=num_new_tokens,
        cache_stats=streamer.get_decode_cache_stats(),
    )


def run_method(args, method, dataset, build_prompt, img_root, benchmark):
    runtime = None
    aggregate = init_aggregate(METHOD_LABELS[method])
    sample_metrics = []
    with cache_hit_decode_prefetch_override(args.model):
        try:
            runtime = _build_runtime(args, method)
            if args.warmup_samples > 0:
                print(f"Warmup {method} cache-hit runtime: {min(int(args.warmup_samples), len(dataset))} samples")
                if runtime["kind"] == "qwen3vlmoe":
                    run_qwen_warmup(runtime, dataset, build_prompt, img_root, args.warmup_samples)
                else:
                    run_deepseek_warmup(runtime, dataset, build_prompt, img_root, args.warmup_samples)

            desc = f"{args.model}-{method}-cache-hit-{benchmark}"
            for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc=desc):
                sample_id = _resolve_sample_id(row, idx)
                source_index = _resolve_source_index(row, idx)
                if runtime["kind"] == "qwen3vlmoe":
                    metric = run_qwen_sample(runtime, row, build_prompt, img_root, args, sample_id, source_index)
                else:
                    metric = run_deepseek_sample(runtime, row, build_prompt, img_root, args, sample_id, source_index)
                sample_metrics.append(metric)
                update_aggregate(aggregate, metric)
        finally:
            _cleanup_runtime(runtime)

    return finalize_aggregate(aggregate), sample_metrics


def main():
    args = parse_args()
    if not (0 < args.sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {args.sample_ratio}")
    if args.keep_rate is not None and not (0 < args.keep_rate <= 1):
        raise ValueError(f"keep_rate must be in (0, 1], got {args.keep_rate}")

    methods = resolve_methods(args.methods)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_cache_env(args)
    dataset_key, benchmark, dataset = load_dataset(args.data_dir)
    dataset = sample_dataset_for_benchmark(dataset, args.sample_ratio, args.sample_seed)
    build_prompt = PROMPT_BUILDERS[dataset_key]
    img_root = resolve_image_root(args.data_dir)

    summary_by_method = {}
    sample_metrics_by_method = {}
    for method in methods:
        print("")
        print("=" * 80)
        print(f"Running {METHOD_LABELS[method]} decode cache-hit benchmark on {benchmark}")
        print("=" * 80)
        summary, sample_metrics = run_method(args, method, dataset, build_prompt, img_root, benchmark)
        summary_by_method[method] = summary
        sample_metrics_by_method[method] = sample_metrics
        print(
            f"{METHOD_LABELS[method]} cache hit: hit_rate={summary['hit_rate']:.4%}, "
            f"hits={summary['hits']}, misses={summary['misses']}, total={summary['total']}"
        )

    payload = {
        "benchmark": benchmark,
        "dataset_key": dataset_key,
        "metric_definition": (
            "Decode-only expert cache hit rate from LinearCache.load_experts(). "
            "Counters are cleared at decode start and aggregate GPU-resident hits plus correctly prefetched hits "
            "vs offloaded misses for AdapMoE/skip/offline/online/TableMoE."
        ),
        "model": args.model,
        "model_name": args.model_name or Path(args.model_path).name or args.model,
        "model_path": args.model_path,
        "methods": methods,
        "method_labels": {method: METHOD_LABELS[method] for method in methods},
        "data_dir": args.data_dir,
        "cache_ratio": args.cache_ratio,
        "keep_rate": args.keep_rate,
        "sample_ratio": args.sample_ratio,
        "sample_seed": args.sample_seed,
        "max_new_tokens": args.max_new_tokens,
        "warmup_samples": args.warmup_samples,
        "cache_dir": args.cache_dir,
        "pca_dir": args.pca_dir,
        "summary_by_method": summary_by_method,
        "sample_metrics_by_method": sample_metrics_by_method,
    }
    if len(methods) == 1:
        payload["method"] = methods[0]
        payload["method_label"] = METHOD_LABELS[methods[0]]
        payload["summary"] = summary_by_method[methods[0]]
        payload["sample_metrics"] = sample_metrics_by_method[methods[0]]

    output_path = output_dir / "decode_cache_hit_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote summary JSON: {output_path}")


if __name__ == "__main__":
    main()
