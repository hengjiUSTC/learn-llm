from peft import PeftModel
import os
import argparse
from trl_finetune import load_config, prepare_model, prepare_tokenizer
from utils import get_logger

logger = get_logger("merge", "info")


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
    args.use_int4 = False
    args.use_int8 = False
    args.disable_flash_attention = True
    args.trust_remote_code = None

    output_dir = os.path.join(args.output_dir, "merged")

    BASE_MODEL = args.model_name
    logger.info("Using base model %s", BASE_MODEL)

    device_map = "auto"
    logger.info("Using Auto device map")
    logger.warning("Make sure you have enough GPU memory to load the model")

    os.makedirs("offload", exist_ok=True)

    tokenizer, _ = prepare_tokenizer(args)
    logger.info("Tokenizer %s", tokenizer)
    logger.info("Loading base model...")
    model = prepare_model(args=args, tokenizer=tokenizer)
    logger.info("Loading Lora model...")

    lora_model = PeftModel.from_pretrained(
        model,
        args.output_dir,
        device_map=device_map,
        offload_folder="offload",
    )

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Merging model...")
    lora_model = lora_model.merge_and_unload()
    logger.info("Merge complete, saving model to %s ...", output_dir)

    lora_model.save_pretrained(output_dir)
    logger.info("Model saved")

    tokenizer.save_pretrained(output_dir)
