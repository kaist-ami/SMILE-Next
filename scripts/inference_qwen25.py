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

import json
from typing import Optional

import fire
from transformers import Seq2SeqTrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer, load_model
from tqdm import tqdm
import torch
import numpy as np

# if is_vllm_available():
#     from vllm import LLM, SamplingParams
#     from vllm.lora.request import LoRARequest


def infer_qwen(
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
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=32,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    # import pdb; pdb.set_trace()
    template_obj.mm_plugin.expand_mm_tokens = True  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    inputs, prompts, labels = [], [], []
    print('get_dataset done')
    for sample in tqdm(dataset_module["train_dataset"], desc='data processing'):
        multi_modal_data = None

        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=skip_special_tokens))
        labels.append(
            tokenizer.decode(
                list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=skip_special_tokens
            )
        )

    print('loading data done')
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    lora_path = '/home/mok/module/ijcv25/models/saves/qwen2.5-7b/moelora/sft_selfinst'
    tokenizer = AutoProcessor.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
    # model = peft_model.merge_and_unload()
    model = peft_model
    
    preds = []
    for e in tqdm(zip(inputs, prompts, labels)):
        input_data, prompt, label = e
        multi_modal_data = input_data["multi_modal_data"]
        prompt_token_ids = input_data["prompt_token_ids"]
        
        
        inputs = torch.tensor([prompt_token_ids], device=model.device)
        attention_mask = torch.ones_like(inputs, dtype=torch.long)

        if multi_modal_data is not None:
            # MoE LoRA is not supported for multi-modal generation in vLLM currently.
            raise NotImplementedError("MoE LoRA is not supported for multi-modal generation in vLLM currently.")

        output = peft_model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=generating_args.max_new_tokens,
            temperature=generating_args.temperature,
            top_p=generating_args.top_p,
            top_k=generating_args.top_k,
            repetition_penalty=generating_args.repetition_penalty,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
        
        
        if label.startswith("Yes.") or label.startswith("No."):
            task = "detection"
            
        elif label.startswith("The laugh type") or label.startswith("The laugh type"):
            task = "classification"
            
        elif label.startswith("The audience laughed") or label.startswith("The person laughed"):
            task = "lreasoning"
            
        else:
            if label.startswith("The humor type"):
                task = "classification"
            else:
                continue
        
        for i in range(28):
            # gate_proj_weight = torch.mean(output.attentions[0][i][0], dim=1)
            # up_proj_weight = torch.mean(output.attentions[0][i][1], dim=1)    
            # down_proj_weight = torch.mean(output.attentions[0][i][2], dim=1)
            
            temp = [e for e in output.attentions]
            temp = [e[i] for e in temp]### layer
            # temp = [[e[0][:, 1:], e[1][:, 1:], e[2][:, 1:], e[3][:, 1:]] for e 
            # in temp[0]]
            
            res = [torch.stack([temp[tok+1][0] for tok in range(len(temp)-1)], dim=1), 
                    torch.stack([temp[tok+1][1] for tok in range(len(temp)-1)], dim=1),
                    torch.stack([temp[tok+1][2] for tok in range(len(temp)-1)], dim=1),
                    torch.stack([temp[tok+1][3] for tok in range(len(temp)-1)], dim=1)]

            # query = torch.mean(res[0], dim=1)
            # key = torch.mean(res[1], dim=1)    
            # value = torch.mean(res[2], dim=1)
            # o = torch.mean(res[3], dim=1)
            
            query = torch.mean(output.attentions[2][i][0], dim=1)
            key = torch.mean(output.attentions[2][i][1], dim=1)    
            value = torch.mean(output.attentions[2][i][2], dim=1)
            o = torch.mean(output.attentions[2][i][3], dim=1)

            # query = output.attentions[0][i][0]#[-1]
            # key = output.attentions[0][i][1]#[-1]
            # value = output.attentions[0][i][2]#[-1]
            # o = output.attentions[0][i][3]#[-1]

            with open(f'./debug{i}.jsonl', "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "task": task,
                    "query_weight_mean": query.cpu().tolist(),
                    "key_weight_mean": key.cpu().tolist(),
                    "value_weight_mean": value.cpu().tolist(),
                    "o_weight_mean": o.cpu().tolist(),
                }, ensure_ascii=False) + "\n")
                
        output = output.sequences
        prompt_length = len(prompt_token_ids)
        output = output[:, prompt_length:]  # remove prompt tokens
        pred = tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)[0]
        preds.append(pred)
        
        # with open(save_name, "a", encoding="utf-8") as f:
        #     # for text, pred, label in zip(prompts, preds, labels):
        #     f.write(json.dumps({"prompt": prompt, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(prompts)} generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    # fire.Fire(infer_qwen)
    infer_qwen(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_name_or_path="./models/saves/qwen2.5-7b/moelora/sft_selfinst",
        dataset="smilenext_text_val",
        dataset_dir="data",
        template="qwen",
        cutoff_len=32768,
        max_samples=None,
        vllm_config="{}",
        save_name="./models/saves/qwen2.5-7b/moelora/sft_selfinst/generated_predictions.jsonl",
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
