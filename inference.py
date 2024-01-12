import yaml
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def perform_inference(config):
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = True

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], trust_remote_code=True
    )

    # test the input
    decode = tokenizer.decode(tokenizer(config["prompt"])["input_ids"])
    print(f"input decode:\n{decode}")
    # Perform the inference
    model_input = tokenizer(config["prompt"], return_tensors="pt").to("cuda")
    print(f"input:\n{model_input}")
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
        print(tokenizer.decode(output, skip_special_tokens=True))


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

    config = load_config(args.config)
    perform_inference(config)
