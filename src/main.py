import argparse
from config import data, model


def main(args):
    model = args.model
    task = args.task

    if task == "SA":
        prompt_instructions = "Can you please tell me the sentiment of this review"
        prompt_querry = "is it positive or nagative?"

        label_map = {
            "5": "positive",
            "4": "positive",
            "3": "positive",
            "2": "negative",
            "1": "negative",
            "0": "negative",
        }
    elif task == "NLI":
        prompt_instructions = ["", " "]
        prompt_querry = [" ", " "]
        label_map = {"": " ", " ": " "}
    # TODO: ask about this
    else:
        print("Task not listed, using empty strings")
        prompt_instructions = [" "]
        prompt_querry = [" "]
        label_map = {"": " "}

    acc = pipeline(prompt_instructions, prompt_querry, label_map, model, task)


if __name__ == "__main__":
    # Args
    #   prompt_instructions: list of strings
    #   prompt_querry: list of strings
    #   label_map: dict
    #   LM_model: string
    #   task: string
    #
    DEFAULT_MODEL = model["DEFAULT_MODEL"]
    DEFAULT_TASK = task["DEFAULT_TASK"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_instructions",
        type=str,
        default="Can you please tell me the sentiment of this review",
    )
    parser.add_argument(
        "--prompt_querry", type=str, default="is it positive or nagative?"
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default={
            "5": "positive",
            "4": "positive",
            "3": "positive",
            "2": "negative",
            "1": "negative",
            "0": "negative",
        },
    )
    parser.add_argument("--LM_model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    args = parser.parse_args()
    main(args)
