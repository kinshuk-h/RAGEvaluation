import os
import gc
import sys
# import signal
import timeit

import torch
import dotenv

dotenv.load_dotenv()

import evaluation.models

MODELS = {
    "BLOOMZ-3B"             : "bigscience/bloomz-3b",
    "BLOOMZ-7.1B"           : "bigscience/bloomz-7b1",
    "LLaMa-2-7B"            : "../models/llama-2-7b-hf/",
    "LLaMa-2-Chat-7B"       : "../models/llama-2-7b-chat-hf/",
    "LLaMa-3-8B"            : "NousResearch/Meta-Llama-3-8B",
    "LLaMa-3-Chat-8B"       : "NousResearch/Meta-Llama-3-8B-Instruct",
    "LLaMa-2-13B"           : "../models/llama-2-13b-hf/",
    "LLaMa-2-Chat-13B"      : "../models/llama-2-13b-chat-hf/",
    "Mistral-Instruct-7B"   : "mistralai/Mistral-7B-Instruct-v0.1",
    "Mistral-Instruct-7B-v2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Gemma-IT-7B"           : "google/gemma-1.1-7b-it",
    "R-Gemma-IT-2B"         : "google/recurrentgemma-2b-it",
    "Zephyr-SFT-7B"         : "alignment-handbook/zephyr-7b-sft-full",
    "Zephyr-Beta-7B"        : "HuggingFaceH4/zephyr-7b-beta",
    "FlanT5-XL-3B"          : "google/flan-t5-xl",
    "GPT-J-6B"              : "eleutherai/gpt-j-6b",
    "Instruct-GPT-J-6B"     : "nlpcloud/instruct-gpt-j-fp16",
    "GPT-3.5-U"             : "openai/azure/gpt-3.5-turbo",
    "GPT-4-U"               : "openai/azure/gpt-4",
}

OPENAI_CREDENTIALS = {
    'org_id' : os.environ['OPENAI_ORG_ID'],
    'api_key': os.environ['OPENAI_API_KEY']
}

AZURE_OPENAI_CREDENTIALS = {
    'endpoint': os.environ['AZURE_OPENAI_ENDPOINT'],
    'api_key': os.environ['AZURE_OPENAI_API_KEY'],
    'deployment': os.environ['AZURE_OPENAI_DEPLOYMENT'],
    'api_version': os.environ['AZURE_OPENAI_API_VERSION']
}

TOKENIZER_ARGS = {
    "*"                : {},
    "BLOOMZ-3B"        : { "padding_side": "left" },
    "BLOOMZ-7.1B"      : { "padding_side": "left" },
    "GPT-J-6B"         : { "padding_side": "left" },
    "Instruct-GPT-J-6B": { "padding_side": "left" },
}

STD_AUTOREGRESSIVE_MODEL_INIT_ARGS = {
    "torch_dtype"        : torch.bfloat16,
    "device_map"         : "auto",
    "low_cpu_mem_usage"  : True,
    "attn_implementation": "sdpa"
}

MODEL_INIT_ARGS = {
    "*"                     : STD_AUTOREGRESSIVE_MODEL_INIT_ARGS,
    "BLOOMZ-3B"             : { "torch_dtype": torch.bfloat16, "device_map": "auto", },
    "BLOOMZ-7.1B"           : { "torch_dtype": torch.bfloat16, "device_map": "auto", },
    "Mistral-Instruct-7B-v2": { **STD_AUTOREGRESSIVE_MODEL_INIT_ARGS, "token": os.environ.get("HF_TOKEN", None) },
    "Gemma-IT-7B"           : { **STD_AUTOREGRESSIVE_MODEL_INIT_ARGS, "token": os.environ.get("HF_TOKEN", None) },
    "R-Gemma-IT-2B"         : { **STD_AUTOREGRESSIVE_MODEL_INIT_ARGS, "token": os.environ.get("HF_TOKEN", None) },
    "FlanT5-XL-3B"          : {},
    "GPT-J-6B"              : {},
    "Instruct-GPT-J-6B"     : { "torch_dtype": torch.bfloat16, "device_map": "auto", },
    "GPT-3.5-U"             : AZURE_OPENAI_CREDENTIALS,
    "GPT-4-U"               : AZURE_OPENAI_CREDENTIALS,
}

def sync_vram():
    while gc.collect() > 0:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

SIZES = " KMGTPEZY"
def format_size(size_bytes, use_si=False, ratio=0.7):
    factor = 1000 if use_si else 1024
    idx, size = 0, size_bytes
    while size >= ratio * factor:
        idx += 1
        size /= factor
    return f"{size:.3f} {SIZES[idx].strip()}{'i' if not use_si and idx > 0 else ''}B"

def show_device_mem_usage():
    print(f" Device | {'Free':>12} | {'Total':>12}")
    print('', '-' * 6, '+', '-' * 12, '+', '-'*12, '', sep='-')
    for device_id in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device_id)
        print(f" cuda:{device_id}", "|", f"{format_size(free):>12}", "|", f"{format_size(total):>12}")

def prepare_model_for_inference(model_key):
    if model_key not in MODELS:
        raise ValueError("Unsupported model specified for inference")

    model_name   = MODELS[model_key]
    model_kwargs = MODEL_INIT_ARGS.get(model_key, MODEL_INIT_ARGS['*'])

    if model_name.lower().startswith("openai"):
        return evaluation.models.OpenAIModelInference(model_name[7:], **model_kwargs)
    else:
        return evaluation.models.HuggingFaceModelInference(model_name, **model_kwargs)

def batchify(*lists, batch_size=8):
    """ Creates batches jointly across lists of iterables.

    Args:
        *lists (list): Lists to batch. If these are of unequal sizes, the minimum size is used.
        batch_size (int, optional): Batch size. Defaults to 8.

    Yields:
        tuple[*list]: Tuple of batches.
    """
    max_len = min(map(len, lists))
    for ndx in range(0, max_len, batch_size):
        yield tuple(
            lst[ndx:min(ndx + batch_size, max_len)]
            for lst in lists
        )

__TIME_UNITS__  = [ 'ms', 's', 'm', 'h', 'd', 'w', 'y' ]
__TIME_RATIOS__ = ( 1000, 60, 60, 24, 7, 52 )

def format_time(time_in_s, ret_type='iter', ratio=0.7):
    time_in_ms = time_in_s * 1e3
    if ret_type == 'iter':
        index = 0
        while index < len(__TIME_RATIOS__) and time_in_ms > (ratio * __TIME_RATIOS__[index]):
            time_in_ms /= __TIME_RATIOS__[index]
            index += 1
        fmtd_time = f"{time_in_ms:.1f}{__TIME_UNITS__[index]}"
    else:
        index = 0
        fmtd_time = ""
        while index < len(__TIME_RATIOS__) and time_in_ms > (ratio * __TIME_RATIOS__[index]):
            mod_time = time_in_ms % __TIME_RATIOS__[index]
            if mod_time != 0:
                fmtd_time = f"{int(mod_time)}{__TIME_UNITS__[index]}" + fmtd_time
            time_in_ms //= __TIME_RATIOS__[index]
            index += 1
        if time_in_ms != 0:
            fmtd_time = f"{int(time_in_ms)}{__TIME_UNITS__[index]}" + fmtd_time
    return fmtd_time

class ModelManager:
    def __init__(self, *args, model_initializer=prepare_model_for_inference, **kwargs) -> None:
        self.model_initializer = model_initializer
        self.args = args
        self.kwargs = kwargs

    def free_resources(self):
        # print("trying to free resources")
        if hasattr(self, 'model') and self.model is not None:
            # print("has model, attempting to delete")
            if hasattr(self.model, 'model'):
                # print("was huggingface model, deleting")
                del self.model.model
            del self.model
        sync_vram()

    def __enter__(self):
        # print(f"manager: reached enter")

        sync_vram()

        # self.past_handler = signal.signal(signal.SIGINT, self.free_resources)
        self.model = self.model_initializer(*self.args, **self.kwargs)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_traceback):
        # print(f"manager: reached exit (due to exception? {exc_val is not None})")

        self.free_resources()
        # signal.signal(signal.SIGINT, self.past_handler)

        if exc_val is not None: raise exc_val

class LogTime:
    def __init__(self, label, verbose=True) -> None:
        self.label = label or "execution completed"
        self.toc = None
        self.tic = None
        self.verbose = verbose

    def __enter__(self):
        if self.verbose: print("[<]", self.label, "...", end = '\n\n')
        self.tic = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_traceback):
        self.toc = timeit.default_timer()
        log_str = f"[>] {self.label}: {format_time(self.toc-self.tic, 'log')}"
        if self.verbose: print()
        print(log_str, file=sys.stderr)
        if self.verbose: print('-' * (3 * len(log_str)), file=sys.stderr)

        if exc_val is not None: raise exc_val

class BatchProgressTimer:
    class Operation:
        def __init__(self, timer, batch, **postfix_kwargs):
            self.timer = timer
            self.postfix_kwargs = postfix_kwargs
            self.time = 0
            self.batch = batch
            self.remaining = self.timer.total - batch

        def __enter__(self):
            self.time = timeit.default_timer()

        def __exit__(self, exc_type, exc_val, exc_traceback):
            end_time = timeit.default_timer()
            self.timer.add_time(end_time - self.time)
            self.timer.progress_bar.set_postfix({
                **self.postfix_kwargs,
                'batch': self.batch, 'total': self.timer.total,
                'cur.': format_time(end_time - self.time, 'iter') + "/it",
                'avg.': format_time(self.timer.avg_time, 'iter') + "/it",
                'eta': format_time(self.timer.avg_time * self.remaining, 'eta')
            })

            if exc_val is not None: raise exc_val

    def __init__(self, progress_bar, total) -> None:
        self.progress_bar = progress_bar
        self.total = total
        self.avg_time = 0

    def timed_operation(self, batch, **postfix_kwargs):
        return self.Operation(self, batch, **postfix_kwargs)

    def add_time(self, time):
        if self.avg_time != 0:
            self.avg_time = 0.9 * self.avg_time + 0.1 * time
        else:
            self.avg_time = time