PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
STOP_MARKER = "\n### Instruction:"


def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def clean_response(text):
    return text.split(STOP_MARKER)[0].strip()
