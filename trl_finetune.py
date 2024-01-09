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
    PreTrainedTokenizerFast,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import concatenate_datasets, Dataset
from accelerate import Accelerator
import bitsandbytes as bnb
import yaml
import dataloader
from utils import get_logger
from dataclasses import dataclass, field
import inspect


logger = get_logger("finetune", "info")

SUPPORTED_FLASH_MODELS = ["llama", "mistral", "falcon", "mixtral", "opt"]


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
    model_dtype: Optional[
        str
    ] = None  # model datatype, only float16 or bfloat16 supported
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
    optimizer: str = "adamw_torch"
    bf16: bool = False
    fp16: bool = False

    # SFTTrainer configuration
    packing: bool = False

    # Additional model configuration
    use_int4: bool = False  # Use int4 precision
    use_int8: bool = False  # Use int8 precision
    disable_lora: bool = False  # Disable LoRA
    disable_flash_attention: bool = False  # Disable flash attention
    all_linear: bool = False  # Use LoRA on all linear layers
    pad_token_id: Optional[int] = None  # End of sequence token ID
    add_eos_token: bool = False  # Add EOS token to tokenizer
    add_bos_token: bool = False  # Add BOS token to tokenizer
    add_pad_token: bool = False  # Add PAD token to tokenizer
    padding_side: Optional[str] = None  # Padding side for tokenizer
    # New field for special tokens
    special_tokens: Dict[str, str] = field(default_factory=lambda: {})
    custom_tokens: List[str] = field(default_factory=list)  # List of custom_tokens

    # Dataset handling
    completion_only: bool = False  # Only use completion loss
    wand_db_project: str = "trl_finetuning"  # Wandb project to use
    prepare_data_path: Optional[str] = None  # dataset cache folder
    datasets: List[DatasetConfig] = field(
        default_factory=list
    )  # List of dataset configurations


def load_config(config_file):
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


def load_and_process_datasets(config: Config):
    # Check if cached dataset exists
    if config.prepare_data_path and os.path.exists(config.prepare_data_path):
        combined_dataset = Dataset.load_from_disk(config.prepare_data_path)
    else:
        # Create a dictionary of handler functions from dataloader.py
        handler_functions = {
            name: func
            for name, func in inspect.getmembers(dataloader, inspect.isfunction)
            if name.startswith("handle_")  # Assuming all handlers start with 'handle_'
        }

        # Process datasets if cache does not exist
        all_datasets = []
        for dataset_config in config.datasets:
            dataset_handler = dataset_config["handler"]
            if dataset_handler in handler_functions:
                handler_function = handler_functions[dataset_handler]
                processed_dataset = handler_function(dataset_config)
                print(processed_dataset[0])
                all_datasets.append(processed_dataset)
            else:
                raise Exception(f"Unsupported dataset handler: {dataset_handler}")

        combined_dataset = concatenate_datasets(all_datasets)
        if len(combined_dataset) > 1:
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


def update_model_special_token(model, tokenizer):
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


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    custom_tokens: Optional[List[str]] = None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    if len(list(special_tokens_dict.keys())) > 0 or custom_tokens is not None:
        logger.info("Resizing tokenizer and embedding...")
        logger.info("Special tokens dict: %s", special_tokens_dict)
        logger.info("Custom tokens: %s", custom_tokens)
    else:
        return False

    num_new_tokens = 0
    if len(list(special_tokens_dict.keys())) > 0:
        num_new_tokens += tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens is not None:
        num_new_tokens += tokenizer.add_tokens(custom_tokens, special_tokens=True)
    logger.info("Number of new tokens: %d", num_new_tokens)

    bos_or_eos_in_special_tokens = (
        "bos_token" in special_tokens_dict or "eos_token" in special_tokens_dict
    )
    if (
        tokenizer.__class__.__name__
        in (
            "LlamaTokenizerFast",
            "CodeLlamaTokenizerFast",
        )
        and bos_or_eos_in_special_tokens
    ):
        logger.info("Tokenizer: update_post_processor")
        tokenizer.update_post_processor()

    return False


def get_model_config(args: Config):
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


def prepare_tokenizer(args: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        add_eos_token=args.add_eos_token,
        add_bos_token=args.add_bos_token,
        use_fast=True,
        truncation=True,
    )

    # THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    # good one for LLama is 18610, mixtral use 0
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(args.pad_token_id)

    if args.padding_side is not None:
        tokenizer.padding_side = args.padding_side

    added_tokens = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=args.special_tokens,
        tokenizer=tokenizer,
        # model=model,
        custom_tokens=args.custom_tokens,
    )

    return tokenizer, added_tokens


def prepare_model(args: Config, tokenizer):
    config = get_model_config(args)
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

    if args.split_model:
        logger.info("Splitting the model across all available devices...")
        kwargs = {"device_map": "auto"}
    else:
        kwargs = {"device_map": None}

    torch_dtype = torch.float32
    if args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    if args.use_int4:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if not args.split_model:
            device_index = Accelerator().process_index
            device_map = {"": device_index}
            kwargs["device_map"] = device_map
        args.use_int8 = False
    elif args.use_int8:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        logger.info("Using no quantization")
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        config=config,
        use_flash_attention_2=use_flash_attention,
        **kwargs,
    )

    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        logger.info("Resize model: %d", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    update_model_special_token(model=model, tokenizer=tokenizer)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()

    args = load_config(args.config)
    print(args)

    train_dataset, validation_dataset = load_and_process_datasets(args)

    # replace_llama_attn(use_full=False)

    os.environ["WANDB_PROJECT"] = args.wand_db_project

    tokenizer, added_tokens = prepare_tokenizer(args)

    logger.info(f"added_tokens: {added_tokens} \ntokenizer: {tokenizer}")

    block_size = args.block_size
    logger.info("Using a block size of %d", block_size)

    model = prepare_model(args=args, tokenizer=tokenizer)

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
        if added_tokens:
            modules_to_save = modules_to_save = ["embed_tokens", "lm_head"]
        else:
            modules_to_save = None
        logger.info(f"modules_to_save: {modules_to_save}")
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
        optim=args.optimizer,
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
        bf16=args.bf16,
        fp16=args.fp16,
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
        packing = args.packing
    logger.info(training_args)
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
