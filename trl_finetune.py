import argparse
import os
from typing import Dict, List, Optional
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
import bitsandbytes as bnb
import yaml
from utils import get_logger
from dataclasses import dataclass, field


logger = get_logger("finetune", "info")

SUPPORTED_FLASH_MODELS = ["llama", "mistral", "falcon", "mixtral", "opt"]
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class DatasetConfig:
    path: str  # Path to the dataset
    split: str  # Dataset split (e.g., 'train', 'test')
    type: dict  # Additional configuration for the dataset


@dataclass
class Config:
    # Training and validation file paths
    val_data_size: float  # Validation data size ratio

    # Model configuration
    model_name: str  # Name of the model
    token: Optional[str] = None  # Authentication token, if required
    split_model: bool = False  # Whether to split the model

    # Model training parameters
    block_size: int = 128  # Size of the blocks used in the model
    lora_rank: int = 64  # LoRA rank
    lora_alpha: Optional[int] = None  # Alpha value for LoRA
    lora_dropout: float = 0.1  # Dropout rate for LoRA
    learning_rate: float = 1e-4  # Learning rate
    lr_scheduler_type: str = "constant"  # Type of learning rate scheduler
    warmup_steps: int = 10  # Number of warmup steps
    weight_decay: float = 0.05  # Weight decay factor
    output_dir: str = "./checkpoints"  # Directory to save model checkpoints
    log_steps: int = 10  # Frequency of logging steps
    eval_steps: int = 10  # Evaluation step frequency
    save_steps: int = 10  # Model saving step frequency
    epochs: float = 1  # Number of training epochs
    batch_size: int = 1  # Training batch size
    gradient_accumulation_steps: int = 1  # Gradient accumulation steps
    gradient_checkpointing: bool = False  # Enable gradient checkpointing
    trust_remote_code: bool = False  # Trust remote code flag
    save_limit: int = 1  # Limit for saving models

    # Additional model configuration
    use_int4: bool = False  # Use int4 precision
    use_int8: bool = False  # Use int8 precision
    disable_lora: bool = False  # Disable LoRA
    disable_flash_attention: bool = False  # Disable flash attention
    all_linear: bool = False  # Use LoRA on all linear layers
    long_lora: bool = False  # Use long LoRA settings
    rope_scale: Optional[float] = None  # ROPE scale value
    pad_token_id: Optional[int] = None  # End of sequence token ID
    add_eos_token: bool = False  # Add EOS token to tokenizer
    add_bos_token: bool = False  # Add BOS token to tokenizer
    add_pad_token: bool = False  # Add PAD token to tokenizer
    padding_side: Optional[str] = None  # Padding side for tokenizer

    # Dataset handling
    train_dataset_ratio: float = 1.0  # Ratio of the training dataset to use
    validation_dataset_ratio: float = 1.0  # Ratio of the validation dataset to use
    completion_only: bool = False  # Only use completion loss
    wand_db_project: str = "trl_finetuning"  # Wandb project to use
    datasets: List[DatasetConfig] = field(
        default_factory=list
    )  # List of dataset configurations


def load_config(config_file):
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


def load_and_process_datasets(config: Config):
    def format_data(row, format_str):
        return (
            format_str.replace("{instruction}", row["instruction"])
            .replace("{input}", row["input"])
            .replace("{output}", row["output"])
        )

    def process_dataset(dataset, format_str):
        return dataset.map(
            lambda row: {"text": format_data(row, format_str)}
        ).remove_columns([c for c in dataset.column_names if c != "text"])

    # Load configuration

    all_datasets = []
    for dataset_config in config.datasets:
        dataset = load_dataset(dataset_config["path"], split=dataset_config["split"])
        format_str = dataset_config["type"]["format"]
        processed_dataset = process_dataset(dataset, format_str)
        all_datasets.append(processed_dataset)

    combined_dataset = concatenate_datasets(all_datasets)

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


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the optimized version that removes the custom_tokens parameter.
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


def get_config(args):
    config_kwargs = {
        "trust_remote_code": True if args.trust_remote_code else None,
        "token": args.token,
    }
    config = AutoConfig.from_pretrained(args.model_name, **config_kwargs)

    config.use_cache = False
    if not args.gradient_checkpointing:
        logger.info("Not using gradient checkpointing")
        config.gradient_checkpointing = False
    else:
        logger.info("Using gradient checkpointing")
        config.gradient_checkpointing = True

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()

    args = load_config(args.config)

    train_dataset, validation_dataset = load_and_process_datasets(args)

    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank * 2
        logger.info(
            "Lora alpha set to None... Setting lora_alpha to %d", args.lora_alpha
        )

    # replace_llama_attn(use_full=False)

    if args.token is None:
        access_token = os.getenv("HF_TOKEN", "")
    else:
        access_token = args.token

    if args.token is None:
        args.token = access_token

    config = get_config(args)
    config_dict = config.to_dict()
    model_type = config_dict["model_type"]

    use_flash_attention = False

    if not args.disable_flash_attention and model_type not in SUPPORTED_FLASH_MODELS:
        logger.info(
            "Model is not llama, mistral, or falcon disabling flash attention..."
        )
    elif args.disable_flash_attention and model_type in SUPPORTED_FLASH_MODELS:
        logger.info(
            "Model is llama, mistral or falcon could be using flash attention..."
        )
    elif not args.disable_flash_attention:
        logger.info("Using flash attention...")
        use_flash_attention = True

    os.environ["WANDB_PROJECT"] = args.wand_db_project

    if args.split_model:
        logger.info("Splitting the model across all available devices...")
        kwargs = {"device_map": "auto"}
    else:
        kwargs = {"device_map": None}

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=access_token,
        trust_remote_code=args.trust_remote_code,
        add_eos_token=args.add_eos_token,
        add_bos_token=args.add_bos_token,
        use_fast=True,
    )

    # THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    # good one for LLama is 18610, mixtral use 0
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(args.pad_token_id)

    if args.padding_side is not None:
        tokenizer.padding_side = args.padding_side

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    block_size = args.block_size
    logger.info("Using a block size of %d", block_size)

    if args.use_int4:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if not args.split_model:
            device_index = Accelerator().process_index
            device_map = {"": device_index}
            kwargs["device_map"] = device_map
        optimizer = "adamw_bnb_8bit"
        args.use_int8 = False
    elif args.use_int8:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        optimizer = "adamw_bnb_8bit"
    else:
        logger.info("Using no quantization")
        bnb_config = None
        optimizer = "adamw_torch"

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=access_token,
        quantization_config=bnb_config,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        config=config,
        use_flash_attention_2=use_flash_attention,
        **kwargs
    )
    added_tokens = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if not args.disable_lora and args.all_linear:
        target_modules = find_all_linear_names(args, model)
        logger.info("Using LORA on all linear layers: %s", target_modules)
        if added_tokens:
            target_modules.pop(target_modules.index("lm_head"))
            logger.info(
                "Removing lm_head from target modules, will use in modules_to_save"
            )
    elif not args.disable_lora:
        target_modules = None
        logger.info("Using LORA on default layers")

    if not args.disable_lora:
        if args.long_lora:
            logger.info("Using long lora settings...")
            modules_to_save = [
                "embed_tokens",
                "input_layernorm",
                "post_attention_layernorm",
                "norm",
            ]

            if added_tokens:
                logger.info("Adding lm_head to modules_to_save")
                modules_to_save.append("lm_head")
        elif added_tokens:
            modules_to_save = modules_to_save = ["embed_tokens", "lm_head"]
        else:
            modules_to_save = None
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        logger.info("Using LORA...")
        if args.use_int4 or args.use_int8:
            logger.info("Preparing model for kbit training...")
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True
                if args.gradient_checkpointing
                else False,
            )

        logger.info("Getting PEFT model...")
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
    else:
        logger.info("Using Full Finetuning")

    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        optim=optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if args.gradient_checkpointing
        else None,
        weight_decay=args.weight_decay,
        report_to="wandb",
        save_total_limit=args.save_limit,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
    )

    if args.completion_only:
        logger.info("Using completion only loss...")
        logger.warning(
            "Make sure to manually set this value in the code to a list of ids"
        )
        response_template = None
        assert response_template is not None, "Response template must be set"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, tokenizer=tokenizer
        )
        packing = False
    else:
        data_collator = None
        packing = None

    # get trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        dataset_text_field="text",
        max_seq_length=block_size,
        tokenizer=tokenizer,
        data_collator=data_collator,
        packing=packing,
    )

    # train
    trainer.train()

    trainer.save_model(args.output_dir)
