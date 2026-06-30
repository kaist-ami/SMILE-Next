# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import fire
from swift_hf_infer import infer_with_swift_hf_engine


def infer_llama3(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
):
    infer_with_swift_hf_engine(
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
        dataset=dataset,
        dataset_dir=dataset_dir,
        template=template,
        cutoff_len=cutoff_len,
        max_samples=max_samples,
        vllm_config=vllm_config,
        save_name=save_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
        pipeline_parallel_size=pipeline_parallel_size,
        image_max_pixels=image_max_pixels,
        image_min_pixels=image_min_pixels,
    )


if __name__ == "__main__":
    infer_llama3(
        model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_name_or_path="./models/saves/llama3-8b/moelora/sft_selfinst",
        dataset="smilenext_text_val",
        dataset_dir="data",
        template="llama3",
        cutoff_len=32768,
        max_samples=None,
        vllm_config="{}",
        save_name="./generated_predictions_urfunny_detection_only.jsonl",
        temperature=0,
        top_p=0.7,
        top_k=50,
        max_new_tokens=1024,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        seed=None,
        pipeline_parallel_size=2,
        image_max_pixels=768 * 768,
        image_min_pixels=32 * 32,
    )
