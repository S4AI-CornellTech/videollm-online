from transformers import HfArgumentParser

from .arguments_live import LiveTrainingArguments, get_args_class
from .live_llama import build_live_llama as build_model_and_tokenizer
from .modeling_live import fast_greedy_generate

def parse_args() -> LiveTrainingArguments:
    # The first pass only reads live_version, so it must tolerate arguments that
    # exist solely on the version-specific subclass (e.g. --embed_mark,
    # --max_num_frames). The second pass parses strictly against that subclass.
    args, *_ = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    args, = HfArgumentParser(get_args_class(args.live_version)).parse_args_into_dataclasses()
    return args