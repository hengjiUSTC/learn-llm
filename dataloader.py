from datasets import load_dataset, concatenate_datasets, Dataset


# General utility functions
def load_dataset_generic(path, split, name):
    return load_dataset(path, split=split, name=name)


def map_dataset(dataset, format_function):
    return dataset.map(lambda row: {"text": format_function(row)}).remove_columns(
        [c for c in dataset.column_names if c != "text"]
    )


# Formatting functions
def format_instruction_data(row, format_str, fields):
    format_dict = {field: row[field] for field in fields}
    return format_str.format(**format_dict)


def format_chat_1_data(row, fields):
    format_str = "<|im_start|>system You are a helpful assistant.<|im_end|>"
    for conv in row[fields]:
        format_str += "\n<|im_start|>user " + conv["input"] + "<|im_end|>"
        format_str += "\n<|im_start|>assistant " + conv["output"] + "<|im_end|>"
    return format_str


# Specific dataset handlers
def handle_instruction_dataset(dataset_config):
    fields = dataset_config["fields"].split(";")
    format_str = dataset_config["format"]
    name = dataset_config["name"] if "name" in dataset_config else None
    path = dataset_config["path"]
    dataset = load_dataset(path, split=dataset_config["split"], name=name)
    formatted_data = map_dataset(
        dataset, lambda row: format_instruction_data(row, format_str, fields)
    )
    return formatted_data


# Formatting functions
def format_instruction_dolly_data(row, format_str, format_str_no_input, fields):
    if row["context"] != "":
        format_dict = {field: row[field] for field in fields}
        return format_str.format(**format_dict)
    else:
        format_dict = {"instruction": row["instruction"], "response": row["response"]}
        return format_str_no_input.format(**format_dict)


# Specific dataset handlers
def handle_instruction_dolly_dataset(dataset_config):
    fields = dataset_config["fields"].split(";")
    format_str = dataset_config["format"]
    format_str_no_input = dataset_config["format_no_input"]
    name = dataset_config["name"] if "name" in dataset_config else None
    path = dataset_config["path"]
    dataset = load_dataset(path, split=dataset_config["split"], name=name)
    formatted_data = map_dataset(
        dataset,
        lambda row: format_instruction_dolly_data(
            row, format_str, format_str_no_input, fields
        ),
    )
    return formatted_data


def handle_chat_1_dataset(dataset_config):
    fields = dataset_config["fields"]
    name = dataset_config["name"] if "name" in dataset_config else None
    path = dataset_config["path"]
    dataset = load_dataset(path, split=dataset_config["split"], name=name)
    formatted_data = map_dataset(dataset, lambda row: format_chat_1_data(row, fields))
    return formatted_data
