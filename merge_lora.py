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
import yaml
from dataclasses import dataclass

logger = get_logger("merge", "info")


@dataclass
class Config:
    base_model: str
    lora_model: str
    output: str


# Function to load configuration from a YAML file
def load_config(file_path: str) -> Config:
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


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

    BASE_MODEL = args.base_model
    logger.info("Using base model %s", BASE_MODEL)

    device_map = "auto"
    logger.info("Using Auto device map")
    logger.warning("Make sure you have enough GPU memory to load the model")

    os.makedirs("offload", exist_ok=True)

    config = AutoConfig.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        args.lora_model,
    )
    logger.info("Tokenizer %s", tokenizer)
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map=device_map,
        offload_folder="offload",
        trust_remote_code=True,
        quantization_config=None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        logger.info("Resize model: %d", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    if (
        hasattr(model, "config")
        and hasattr(model.config, "bos_token_id")
        and model.config.bos_token_id
        and model.config.bos_token_id != tokenizer.bos_token_id
    ):
        model.config.bos_token_id = tokenizer.bos_token_id

    if (
        hasattr(model, "config")
        and hasattr(model.config, "eos_token_id")
        and model.config.eos_token_id
        and model.config.eos_token_id != tokenizer.eos_token_id
    ):
        model.config.eos_token_id = tokenizer.eos_token_id

    logger.info("Loading Lora model...")

    lora_model = PeftModel.from_pretrained(
        model,
        args.lora_model,
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
