import argparse
import os
from transformers import (
    TrainingArguments,
)
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import DPOTrainer
from peft import LoraConfig
import torch
from trl_finetune import load_config, prepare_model, prepare_tokenizer
from utils import get_logger
import bitsandbytes as bnb

SUPPORTED_FLASH_MODELS = ["llama", "mistral", "falcon", "mixtral", "opt"]

logger = get_logger("finetune", "info")


# def chatml_format(example):
#     # Format instruction
#     prompt = f"<|startoftext|>[INST]{example['system']} {example['question']}[/INST]"
#     # prompt = f"<|startoftext|>[INST]{example['prompt']}[/INST]"

#     # Format chosen answer
#     chosen = example["chosen"]

#     # Format rejected answer
#     rejected = example["rejected"]

#     return {
#         "prompt": prompt,
#         "chosen": chosen + "<|endoftext|>",
#         "rejected": rejected + "<|endoftext|>",
#     }


def chatml_format(example):
    # Format system
    if len(example["system"]) > 0:
        message = {"role": "system", "content": example["system"]}
        system = f"<|im_start|>system\n{example['system']}<|im_end|>"
    else:
        system = ""

    # Format instruction
    prompt = f"<|im_start|>user\n{example['question']}<|im_end|>"

    # Format chosen answer
    chosen = example["chosen"] + "<|im_end|>\n"

    # Format rejected answer
    rejected = example["rejected"] + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def loaddata(config):
    if config.prepare_data_path and os.path.exists(config.prepare_data_path):
        logger.info("load datasets from disk")
        combined_dataset = Dataset.load_from_disk(config.prepare_data_path)
    else:
        logger.info("load datasets from hub")
        all_datasets = []
        for dataset_config in config.datasets:
            # Load dataset
            name = dataset_config["name"] if "name" in dataset_config else None
            path = dataset_config["path"]
            dataset = load_dataset(path, split=dataset_config["split"], name=name)

            # Save columns
            original_columns = dataset.column_names

            # Format dataset
            dataset = dataset.map(chatml_format, remove_columns=original_columns)
            print(dataset[0])
            all_datasets.append(dataset)
        combined_dataset = concatenate_datasets(all_datasets)

        logger.info("shuffle merged datasets")
        combined_dataset = combined_dataset.shuffle()
        if config.prepare_data_path:
            # Save combined dataset to disk
            combined_dataset.save_to_disk(config.prepare_data_path)

    # Split data
    split_dataset = combined_dataset.train_test_split(
        test_size=config.val_data_size,
        shuffle=True,
    )

    return split_dataset["train"], split_dataset["test"]


def find_all_linear_names(args, model, add_lm_head=True):
    cls = (
        bnb.nn.Linear4bit
        if args.use_int4
        else (bnb.nn.Linear8bitLt if args.use_int8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if add_lm_head and not "lm_head" in lora_module_names:
        logger.info("Adding lm_head to lora_module_names")
        lora_module_names.add("lm_head")

    return list(lora_module_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()

    args = load_config(args.config)

    os.environ["WANDB_PROJECT"] = args.wand_db_project

    train_dataset, validation_dataset = loaddata(args)
    tokenizer, _ = prepare_tokenizer(args)
    logger.info(f"tokenizer: {tokenizer}")
    model = prepare_model(args, tokenizer)
    ref_model = prepare_model(args, tokenizer)

    target_modules = find_all_linear_names(args, model, add_lm_head=False)

    # Training arguments
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        optim=args.optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="wandb",
        save_total_limit=args.save_limit,
        bf16=args.bf16,
        fp16=args.fp16,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_prompt_length=args.block_size,
        max_length=args.block_size * 2,
        max_target_length=args.block_size,
    )

    dpo_trainer.train()

    dpo_trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
