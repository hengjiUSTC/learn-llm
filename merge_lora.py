import torch
from peft import PeftModel
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import argparse
from utils import get_logger
import transformers
from typing import Dict, Optional
import json
import yaml
from dataclasses import dataclass, field

logger = get_logger("merge", "info")

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class Config:
    base_model: str
    lora_model: str
    output: str
    cpu: Optional[bool] = field(default=False)
    context_size: Optional[int] = field(default=None)
    custom_tokens: Optional[str] = field(default=None)
    pad_token_id: Optional[int] = field(default=None)


# Function to load configuration from a YAML file
def load_config(file_path: str) -> Config:
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    if len(list(special_tokens_dict.keys())) > 0:
        logger.info("Resizing tokenizer and embedding...")
        logger.info("Special tokens dict: %s", special_tokens_dict)
    else:
        return False
    num_new_tokens = len(list(special_tokens_dict.keys()))
    logger.info(
        "Number of new tokens: %d, Special tokens dict: %s",
        num_new_tokens,
        special_tokens_dict,
    )
    tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return True


if __name__ == "__main__":
    # Command line argument to specify the path of the YAML configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    # Load configuration
    args = load_config(args.config)

    args.output = os.path.realpath(args.output)

    LORA_WEIGHTS = os.path.realpath(args.lora_model)

    if args.base_model is not None:
        BASE_MODEL = args.base_model
        logger.info("Using base model %s", BASE_MODEL)
    else:
        adapter_config_path = os.path.join(LORA_WEIGHTS, "adapter_config.json")
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        BASE_MODEL = adapter_config["base_model_name_or_path"]
        logger.info("Base model not given, using %s", BASE_MODEL)

    if args.cpu:
        device_map = {"": "cpu"}
        logger.info("Using CPU")
        logger.warning("This will be slow, use GPUs with enough VRAM if possible")
    else:
        device_map = "auto"
        logger.info("Using Auto device map")
        logger.warning("Make sure you have enough GPU memory to load the model")

    os.makedirs("offload", exist_ok=True)

    config = AutoConfig.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=args.context_size
        if args.context_size is not None
        else config.max_position_embeddings,
    )

    # THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    # good one for LLama is 18610
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(args.pad_token_id)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device_map,
        offload_folder="offload",
        trust_remote_code=True,
        quantization_config=None,
    )

    added_tokens = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    logger.info("Loading Lora model...")

    lora_model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device_map,
        offload_folder="offload",
    )

    os.makedirs(args.output, exist_ok=True)
    logger.info("Merging model...")
    lora_model = lora_model.merge_and_unload()
    logger.info("Merge complete, saving model to %s ...", args.output)

    lora_model.save_pretrained(args.output)
    logger.info("Model saved")

    tokenizer.save_pretrained(args.output)
