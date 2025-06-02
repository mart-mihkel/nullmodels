if __name__ == "__main__":
    import os
    import argparse
    import tempfile

    from estnltk import Text
    from estnltk.converters import json_to_text
    from estnltk.converters.label_studio.labelling_tasks import DiffTaggingTask
    from estnltk.converters.label_studio.labelling_configurations import (
        DiffTaggingConfiguration,
    )

    parser = argparse.ArgumentParser(description="Create labelstudio diff tagging task")
    parser.add_argument(
        "--text-dir",
        "-d",
        type=str,
        help="Path to directory with json serialized texts",
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=os.path.join(tempfile.gettempdir(), "diff_tag", "tasks"),
        help="Out dir",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.text_dir):
        parser.error(f"{args.text_dir} is not a valid directory")

    in_texts: list[Text] = []
    in_txt_files = os.listdir(args.text_dir)
    for in_txt_file in in_txt_files:
        in_txt_path = os.path.join(args.text_dir, in_txt_file)
        in_texts.append(json_to_text(file=in_txt_path))

    conf = DiffTaggingConfiguration(
        layer_a="ner",
        layer_b="estbertner",
        class_labels=["PER", "ORG", "LOC"],
        header="Choose text with correct annotation",
    )

    task = DiffTaggingTask(configuration=conf, label_attribute="nertag")

    os.makedirs(args.out_dir, exist_ok=True)
    task.export_data(in_texts, os.path.join(args.out_dir, "task.json"))
    with open(os.path.join(args.out_dir, "labelling_interface.xml"), "w") as f:
        f.write(task.interface_file)
