import argparse
import random
import numpy as np
import torch
from sampling import autoregressive_generate, speculative_generate, speculative_generate_multi
from ngram_assisted import OneLevelNGramStorage, NGramStorage, ngram_assisted_speculative_generate
from utils.logits_processor import GreedyProcessor, MultinomialProcessor, TopKProcessor, NucleusProcessor, TopKNucleusProcessor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
import time
import os
from termcolor import colored
from copy import copy

import pandas as pd
from datetime import datetime
import json

# this class is modified from infer.py
class Experiment:
    def __init__(
            self, 
            device: str = "cuda",
            target_model = "Qwen/Qwen3-8B",
            drafter_model = "Qwen/Qwen3-0.6B",
            samples = 20, 
            gamma: int = 5,
            trial: int = 5,
            gen_len: int = 100,
            prune_width: int = 0,
            comments: str = "experiment", 
            result_file_name: str = "basic_test",
            configs = {}
        ):
        print(
            colored("Speculative Decoding", "red"),
            colored("CLI", on_color="on_red", color="white"),
            "\n",
        )
        self.device = device
        self.target_model_name = target_model
        self.drafter_model_name = drafter_model
        self.samples = samples
        self.comments = comments
        self.file_name = result_file_name
        self.configs = configs

        self.gamma = gamma
        self.trial = trial
        self.prune_width = prune_width
        self.gen_len = gen_len
        self.debug = False
        self.spec = False
        self.spec_multi = True
        self.dr = False
        self.cache = False
        self.target_gen = False
        # Ngram Assisted Generation
        self.ngram_gen = False
        self.ngram = None
        self.top_k_filler = 3
        self.ngram_n = 3
        self.reset_in_between = True
        
        self.chat = True # If using a chat instructed model, set to True
        
        self.processors = {
            "greedy": {
                "processor": GreedyProcessor,
                "building_args": {"temperature": float},
            },
            "multinomial": {
                "processor": MultinomialProcessor,
                "building_args": {"temperature": float},
            },
            "topk": {
                "processor": TopKProcessor,
                "building_args": {"temperature": float, "top_k": int},
            },
            "nucleus": {
                "processor": NucleusProcessor,
                "building_args": {"temperature": float, "top_p": float},
            },
            "topknucleus": {
                "processor": TopKNucleusProcessor,
                "building_args": {"temperature": float, "top_k": int, "top_p": float},
            },
        }
        self.selected_processor = {
            "name": "greedy",
            "processor": GreedyProcessor,
            "args": {"temperature": 1.0},
        }
        self.processor = TopKProcessor(temperature=1.0, top_k=10)

        self._load_models()
        self._run()

    def _load_models(self):
        # Target model
        target_model = self.target_model_name
        # target_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        target_quantize = QuantoConfig(weights="int8")  # QuantoConfig(weights="int8")  None
        
        # Drafter model
        drafter_model = self.drafter_model_name
        drafter_quantize = QuantoConfig(weights="int8")  # QuantoConfig(weights="int8") None

        print(colored("Target model:", on_color="on_yellow"), target_model)
        print(colored("Drafter model:", on_color="on_yellow"), drafter_model)
        print(colored("Loading models...", "light_grey"))

        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            # quantization_config=target_quantize,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.target.eval()

        tokenizer_name = target_model
        if tokenizer_name != target_model:
            print(colored("Warning: Tokenizer is different from target model. Use with caution.", "red"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            # quantization_config=drafter_quantize,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.drafter.eval()
        
        self.ngram = NGramStorage(n=3, vocab_size=self.target.config.vocab_size)
        
        self.end_tokens = [self.tokenizer.eos_token_id]
        if self.tokenizer.convert_tokens_to_ids("<|eot_id|>"):  # "<|eot_id|>" is the end of turn token for Llama model.
            self.end_tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>")) 
        

    def _infer(self, prefix: str):
        if self.chat:
            prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": prefix}], add_generation_prompt=True, tokenize=False)
            
        tokenized = self.tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()
        
        if self.reset_in_between:
            self.ngram.reset()
        
        spec_throughput = 0.0
        base_throughput = 0.0
        drafter_throughput = 0.0

        if self.spec:
            self._set_seed(42)
            spec_start_time = time.time()
            output_ids, drafts_accepted, drafts_speculated = speculative_generate(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
            )
            accept_rate = drafts_accepted/drafts_speculated
            spec_end_time = time.time()
            spec_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Speculative ==========", "green"))
            print(colored("Out:", "green"), spec_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "green"))
            spec_throughput = len(output_ids) / (spec_end_time - spec_start_time)
            print(colored(f"Throughput: {spec_throughput:.1f} tokens/s", "green"))
            print(colored("========== Speculative ==========", "green"))
            spec_result = [len(spec_output), (spec_end_time - spec_start_time), 
                drafts_accepted, drafts_speculated]
        
        if self.spec_multi:
            self._set_seed(42)
            spec_multi_start_time = time.time()
            output_ids, drafts_accepted, drafts_speculated = speculative_generate_multi(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                trial=self.trial,
                prune_width=self.prune_width,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
            )
            accept_rate = drafts_accepted/drafts_speculated
            spec_multi_end_time = time.time()
            spec_multi_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Speculative (Multi) ==========", "green"))
            print(colored("Out:", "green"), spec_multi_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "green"))
            spec_multi_throughput = len(output_ids) / (spec_multi_end_time - spec_multi_start_time)
            print(colored(f"Throughput: {spec_multi_throughput:.1f} tokens/s", "green"))
            print(colored("========== Speculative (Multi) ==========", "green"))
            spec_multi_result = [len(spec_multi_output), (spec_multi_end_time - spec_multi_start_time), 
                drafts_accepted, drafts_speculated]


        # if self.ngram_gen:
        #     self._set_seed(42)
        #     ngram_start_time = time.time()
        #     output_ids, accept_rate = ngram_assisted_speculative_generate(
        #         tokenized,
        #         self.ngram,
        #         self.target,
        #         tokenizer=self.tokenizer,
        #         gamma=self.gamma,
        #         filler_top_k=self.top_k_filler,
        #         logits_processor=self.processor,
        #         max_gen_len=self.gen_len,
        #         eos_tokens_id=self.end_tokens,
        #         debug=self.debug,
        #         use_cache=self.cache,
        #         first_target=True,
        #         stop_if_unknown=True,
        #     )
        #     ngram_end_time = time.time()
        #     ngram_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        #     print(colored("========== Ngram Assisted ==========", "yellow"))
        #     print(colored("Out:", "yellow"), ngram_output)
        #     print(colored(f"Acceptance rate: {accept_rate:.3f}", "yellow"))
        #     ngram_throughput = len(output_ids) / (ngram_end_time - ngram_start_time)
        #     print(colored(f"Throughput: {ngram_throughput:.1f} tokens/s", "yellow"))
        #     print(colored("========== Ngram Assisted ==========", "yellow"))
        #     if self.spec and ngram_throughput > 0.0:
        #         print(colored(f"Throughput increase: {((spec_throughput / ngram_throughput)) * 100:.1f}%", "magenta"))

        if self.target_gen:
            self._set_seed(42)
            start_time = time.time()
            output_ids = autoregressive_generate(
                tokenized,
                self.target,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )
            end_time = time.time()
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("=========== Target AR ===========", "blue"))
            print(colored("Out:", "blue"), output)
            base_throughput = len(output_ids) / (end_time - start_time)
            print(colored(f"Throughput: {base_throughput:.1f} tokens/s", "blue"))
            print(colored("=========== Target AR ===========", "blue"))
            if self.spec and base_throughput > 0.0:
                print(colored(f"Throughput increase: {((spec_throughput / base_throughput)) * 100:.1f}%", "magenta"))
            target_ar_result = [len(output), (end_time - start_time), 
                0, 0] # zeros for consistency, can be ignored

        # if self.dr:
        #     self._set_seed(42)
        #     output_ids = autoregressive_generate(
        #         tokenized,
        #         self.drafter,
        #         use_cache=self.cache,
        #         max_gen_len=self.gen_len,
        #         eos_tokens_id=self.end_tokens,
        #         logits_processor=self.processor,
        #         debug=self.debug,
        #     )
        #     output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        #     print(colored("========== Drafter AR ==========", "cyan"))
        #     drafter_throughput = len(output_ids) / (end_time - start_time)
        #     print(colored("Out:", "cyan"), output)
        #     print(colored(f"Throughput: {drafter_throughput:.1f} tokens/s", "cyan"))
        #     print(colored("========== Drafter AR ==========", "cyan"))
        return {
            # "speculative": spec_result,
            "speculative_multi": spec_multi_result,
            # "target_ar": target_ar_result
        }

    def _run(self):
        df = pd.read_pickle("short_prompts10k.pkl")
        # collect result
        result = dict()
        for row in df.sample(n=self.samples, random_state=15642)['prompt']:
            print("row: " + row)
            outcome = self._infer(row)
            if not result: # empty
                result = outcome
            else:
                for model,v in result.items():
                    outcome_v = outcome[model]
                    v = [v[0] + outcome_v[0], v[1] + outcome_v[1], 
                        v[2] + outcome_v[2], v[3] + outcome_v[3]]
                    result[model] = v
        # calculate throughput
        new_result = dict()
        for k, v in result.items():
            model_result = dict()
            model_result['num_token'] = v[0]
            model_result['time'] = v[1]
            if v[0] == 0 or v[1] == 0:
                model_result['throughput'] = 0
            else:
                model_result['throughput'] = v[0]/v[1]
            if v[2] == 0 or v[3] == 0:
                model_result['acceptance rate'] = 0
            else:
                model_result['acceptance rate'] = v[2]/v[3]
            new_result[k] = model_result
        experiment_result = {
            "time": datetime.now().isoformat(),
            "drafter_model": self.drafter_model_name,
            "target_model": self.target_model_name,
            "result": new_result,
            "comments": self.comments,
            "configs": self.configs
        }
        print(experiment_result["time"])
        # append to previous results
        try:
            with open(self.file_name + ".json", "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []  # Initialize if file doesn't exist

        data.append(experiment_result)

        with open(self.file_name + ".json", "w") as file:
            json.dump(data, file, indent=4)

        
            
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Speculative Decoding CLI")
    # parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    # args = parser.parse_args()
    with open('experiment_configs.json') as f:
        configs = json.load(f)  # Returns a list containing your config object
    i = 1
    for experiment in configs:
        print(colored("Starting experiment: ", on_color="on_red"), i)
        experiment['configs'] = experiment.copy()
        Experiment(**experiment)
        i+=1
    # Experiment()


