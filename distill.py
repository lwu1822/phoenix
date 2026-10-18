"""Create Veri SFT examples from prompts using a temporary Veri deployment."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def api_key():
    if os.environ.get("VERI_API_KEY"):
        return os.environ["VERI_API_KEY"]
    config = Path.home() / ".config/veri/config.toml"
    data = tomllib.loads(config.read_text())
    profile = data.get("profiles", {}).get(data.get("active_profile", "default"), {})
    if profile.get("api_key"):
        return profile["api_key"]
    raise RuntimeError("Set VERI_API_KEY from your Veri account before running distill.py")


def chat(deployment_id, key, prompt):
    payload = json.dumps({
        "model": "any",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    request = Request(
        f"https://api.veri.studio/v1/deployments/{deployment_id}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"]


async def answer(deployment_id, key, prompt, semaphore):
    async with semaphore:
        for attempt in range(12):
            try:
                text = await asyncio.to_thread(chat, deployment_id, key, prompt)
                if not text:
                    raise ValueError("Teacher returned an empty answer")
                return {"messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ]}
            except HTTPError as exc:
                if exc.code not in (429, 503) or attempt == 11:
                    raise
                delay = exc.headers.get("Retry-After")
                await asyncio.sleep(float(delay) if delay and delay.isdigit() else min(2 ** attempt, 30))
        raise RuntimeError("Teacher unavailable after retries")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment_id")
    parser.add_argument("--prompts", default="train_prompts.jsonl")
    parser.add_argument("--output", default="train.jsonl")
    args = parser.parse_args()
    prompts = [json.loads(line)["prompt"] for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if len(prompts) != 500 or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Expected exactly 500 nonempty string prompts")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; refusing to overwrite distillation data")
    semaphore = asyncio.Semaphore(16)
    key = api_key()
    tasks = [asyncio.create_task(answer(args.deployment_id, key, prompt, semaphore)) for prompt in prompts]
    rows = await asyncio.gather(*tasks)
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(output)
    print(f"Wrote {len(rows)} examples to {output}")


if __name__ == "__main__":
    asyncio.run(main())
