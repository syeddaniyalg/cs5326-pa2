from datasets import load_dataset
from tqdm import tqdm


HUMAN_MARKER = "\n\nHuman:"
ASSISTANT_MARKER = "\n\nAssistant:"


def parse_hh_single_turn(text):
    if text.count(HUMAN_MARKER) != 1 or text.count(ASSISTANT_MARKER) != 1:
        return None
    head, response = text.split(ASSISTANT_MARKER)
    if not head.startswith(HUMAN_MARKER):
        return None
    return head[len(HUMAN_MARKER):].strip(), response.strip()


def make_preference(instruction, chosen, rejected):
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    return {"instruction": instruction, "chosen": chosen, "rejected": rejected}


def convert_sft_row(row):
    messages = row["messages"]
    if len(messages) < 2:
        return None
    user, assistant = messages[0], messages[1]
    if user["role"] != "user" or assistant["role"] != "assistant":
        return None

    instruction = user["content"].strip()
    response = assistant["content"].strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response}


def convert_hh_row(row):
    chosen = parse_hh_single_turn(row["chosen"])
    rejected = parse_hh_single_turn(row["rejected"])
    if chosen is None or rejected is None:
        return None
    if chosen[0] != rejected[0]:
        return None
    return make_preference(chosen[0], chosen[1], rejected[1])


def has_single_turn_roles(messages):
    return len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant"


def convert_ultrafeedback_row(row):
    chosen, rejected = row["chosen"], row["rejected"]
    if not has_single_turn_roles(chosen) or not has_single_turn_roles(rejected):
        return None

    instruction = chosen[0]["content"].strip()
    if rejected[0]["content"].strip() != instruction:
        return None
    return make_preference(instruction, chosen[1]["content"].strip(), rejected[1]["content"].strip())


SOURCES = {
    "sft": {"name": "HuggingFaceH4/ultrachat_200k", "train_split": "train_sft", "val_split": "test_sft", "convert": convert_sft_row},
    "hh": {"name": "Anthropic/hh-rlhf", "train_split": "train", "val_split": "test", "convert": convert_hh_row},
    "ultrafeedback": {"name": "HuggingFaceH4/ultrafeedback_binarized", "train_split": "train_prefs", "val_split": "test_prefs", "convert": convert_ultrafeedback_row},
}


def collect_examples(dataset, convert, count, seed, description):
    examples = []
    bar = tqdm(total=count, desc=description)

    for row in dataset.shuffle(seed=seed):
        example = convert(row)
        if example is None:
            continue
        examples.append(example)
        bar.update(1)
        if len(examples) == count:
            break

    bar.close()
    return examples


def build_task_splits(task, train_size, val_size, seed):
    source = SOURCES[task]
    train_dataset = load_dataset(source["name"], split=source["train_split"])
    val_dataset = load_dataset(source["name"], split=source["val_split"])

    train_rows = collect_examples(train_dataset, source["convert"], train_size, seed, f"{task} train")
    val_rows = collect_examples(val_dataset, source["convert"], val_size, seed, f"{task} val")
    return train_rows, val_rows
