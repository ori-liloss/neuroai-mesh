"""Small-model LoRA experiments for a side machine (for example a laptop GPU).

The big run stays on the main NVIDIA PC. This script answers cheap questions before the big run
spends days of GPU time: which learning rate, which data mix, does a new batch of verified rows
help or hurt. Results transfer roughly, not exactly, to the big model.

Data: a JSONL file of {"messages": [{"role": ..., "content": ...}, ...]} rows, e.g. the verified
rows exported by `mesh.py jobs export`. A fixed slice is held out for eval.

Example, on a side computer that already has a small model folder:
  python -u side_train.py --base C:/models/small-base --data jobs/verified.jsonl --out runs/lr1e-4 --learning-rate 1e-4 --max-steps 300

Packing: several conversations share one forward pass; position_ids restart per conversation and
attention_mask is omitted, so transformers builds a block-diagonal causal mask. The KV cache must
stay off for that, which this script enforces.
"""

import argparse
import json
import os
from pathlib import Path
import random
import stat
import sys

# Same defaults as mesh.py: folders next to this repo, or NEURO_GPU_LOCK / NEURO_AI_DATA.
_HERE = Path(__file__).resolve().parent
MAIN_LOCK = Path(os.environ.get("NEURO_GPU_LOCK") or (_HERE.parent / "train_local" / "gpu.lock"))
LIVE = Path(os.environ.get("NEURO_AI_DATA") or (_HERE.parent / "neuro-ai-data")) / "training"


def busy(path):
    """FileNotFoundError means free. Any other error, or an existing file, means busy."""
    try:
        os.stat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def inside_live(path: Path) -> bool:
    try:
        path.resolve().relative_to(LIVE.resolve())
    except ValueError:
        return False
    except OSError:
        return True
    return True


def read_rows(path):
    try:
        if os.stat(path).st_size > 64_000_000:
            raise SystemExit("The data file is too large for a side run.")
    except FileNotFoundError:
        raise SystemExit("The data file is missing. This side run was not started.")
    except OSError:
        raise SystemExit("The data file could not be read. This side run was not started.")
    out = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            if len(line) > 200_000:
                raise SystemExit("The data file has a line that is too long. This side run was not started.")
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = row.get("messages") if isinstance(row, dict) else None
            if (isinstance(messages, list) and messages
                    and all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages)
                    and any(m.get("role") == "assistant" for m in messages)):
                out.append(messages)
    return out


def ids(tokenizer, messages, prompt=False):
    return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=prompt,
                                              return_dict=False))


def encode(tokenizer, messages, limit):
    """Loss only on assistant turns, same rule as ml/training/train.py."""
    full = ids(tokenizer, messages)
    if len(full) > limit:
        return None
    labels = [-100] * len(full)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = ids(tokenizer, messages[:index], prompt=True)
        response = ids(tokenizer, messages[:index + 1])
        if len(response) <= len(prefix) or response[:len(prefix)] != prefix or full[:len(response)] != response:
            return None
        labels[len(prefix):len(response)] = full[len(prefix):len(response)]
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": full, "labels": labels}


def pack(samples, budget, seed):
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    bins, sizes = [], []
    for index in order:
        size = len(samples[index]["input_ids"])
        start = max(0, len(sizes) - 64)
        for slot in range(start, len(sizes)):
            if sizes[slot] + size <= budget:
                bins[slot].append(samples[index])
                sizes[slot] += size
                break
        else:
            bins.append([samples[index]])
            sizes.append(size)
    return bins


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Local folder of a small base model. A remote name is refused.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--target-examples", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--four-bit", action="store_true", help="QLoRA (needs bitsandbytes). Default is bf16/fp16 LoRA.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if busy(MAIN_LOCK):
        raise SystemExit("The main training lock is held or unreadable. This side run was not started.")
    if inside_live(args.data) or inside_live(args.out):
        raise SystemExit("Refusing a path inside the live training directory.")
    if not 0 < args.learning_rate <= 0.01 or not 0 < args.eval_fraction < 0.5 or args.max_length < 256:
        parser.error("Invalid settings.")
    if len(args.base) > 240 or any(ord(ch) < 32 for ch in args.base) or ".." in Path(args.base).parts:
        raise SystemExit("The base folder could not be read. This side run was not started.")
    try:
        base_info = Path(args.base).stat()
    except FileNotFoundError:
        raise SystemExit("The base folder is missing. This side run was not started.")
    except OSError:
        raise SystemExit("The base folder could not be read. This side run was not started.")
    if not stat.S_ISDIR(base_info.st_mode):
        raise SystemExit("The base folder is missing. This side run was not started.")
    lock = args.out.with_name(args.out.name + ".lock")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        with lock.open("x", encoding="ascii") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise SystemExit("Another side run holds the lock. This side run was not started.")
    except OSError:
        raise SystemExit("The side lock could not be read. This side run was not started.")
    try:
        if busy(MAIN_LOCK):
            raise SystemExit("The main training lock is held or unreadable. This side run was not started.")

        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

        if not torch.cuda.is_available():
            raise SystemExit("This script needs a CUDA GPU. On the Mac use mlx_lm.lora instead (see README).")
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        conversations = read_rows(args.data)
        samples = [s for s in (encode(tokenizer, m, args.max_length) for m in conversations) if s]
        if len(samples) < 40:
            raise SystemExit("There are fewer than 40 usable rows. This side run was not started.")
        random.Random(args.seed).shuffle(samples)
        n_eval = max(8, int(len(samples) * args.eval_fraction))
        eval_bins = pack(samples[:n_eval], args.max_length, args.seed + 1)
        train_bins = pack(samples[n_eval:], args.max_length, args.seed)
        accum = max(1, round(args.target_examples / ((len(samples) - n_eval) / len(train_bins))))
        print(json.dumps({"rows": len(conversations), "usable": len(samples), "train_bins": len(train_bins),
                          "eval_bins": len(eval_bins), "accumulation": accum}), flush=True)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = {"dtype": dtype, "device_map": {"": 0}, "attn_implementation": "sdpa"}
        if args.four_bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, **kwargs)
        model.config.use_cache = False
        if args.four_bit:
            model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))

        def collate(features):
            input_ids, labels, positions = [], [], []
            for group in features:
                for row in group:
                    input_ids.extend(row["input_ids"])
                    part = list(row["labels"])
                    part[0] = -100
                    labels.extend(part)
                    positions.extend(range(len(row["input_ids"])))
            return {"input_ids": torch.tensor([input_ids]), "labels": torch.tensor([labels]),
                    "position_ids": torch.tensor([positions]), "use_cache": False}

        trainer = Trainer(
            model=model, train_dataset=train_bins, eval_dataset=eval_bins, data_collator=collate,
            processing_class=tokenizer,
            args=TrainingArguments(
                output_dir=str(args.out), num_train_epochs=args.epochs, max_steps=args.max_steps,
                per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=accum,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.03,
                max_grad_norm=1.0, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
                optim="paged_adamw_8bit" if args.four_bit else "adamw_torch",
                eval_strategy="steps", eval_steps=50, save_strategy="no", logging_steps=10,
                prediction_loss_only=True, remove_unused_columns=False, report_to="none", seed=args.seed,
            ),
        )
        trainer.train()
        metrics = trainer.evaluate()
        metrics.update({"learning_rate": args.learning_rate, "base": args.base, "data": str(args.data)})
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        trainer.save_model(str(args.out / "adapter"))
        print(json.dumps(metrics), flush=True)
    finally:
        try:
            if lock.read_text(encoding="ascii").strip() == str(os.getpid()):
                lock.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
