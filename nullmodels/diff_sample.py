import os
import sqlite3
import warnings

from collections import defaultdict

from estnltk import Layer, Text
from estnltk.converters.label_studio.labelling_tasks import DiffTaggingTask
from estnltk.converters.label_studio.labelling_configurations import (
    DiffTaggingConfiguration,
)

type TextDiffRecord = tuple[
    str,
    int | None,
    int | None,
    str | None,
    int | None,
    int | None,
    str | None,
]


def __attach_diff_annotations(
    annotations: list[TextDiffRecord],
    layer_a: str,
    layer_b: str,
    label_attribute: str,
) -> list[Text]:
    """
    Create new texts with just the sampled annotations.
    """
    annt_grouped = defaultdict(list)
    for annt in annotations:
        text, annt = annt[0], annt[1:]
        annt_grouped[text].append(annt)

    res = []
    for text, annt in annt_grouped.items():
        new_text = Text(text)

        new_a = Layer(layer_a, attributes=[label_attribute], ambiguous=True)
        new_b = Layer(layer_b, attributes=[label_attribute], ambiguous=True)

        new_text.add_layer(new_a)
        new_text.add_layer(new_b)

        for a_start, a_end, a_annt, b_start, b_end, b_annt in annt:
            if a_start is not None and a_end is not None and a_annt:
                attr = {label_attribute: a_annt}
                new_a.add_annotation((a_start, a_end), attribute_dict=attr)

            if b_start is not None and b_end is not None and b_annt:
                attr = {label_attribute: b_annt}
                new_b.add_annotation((b_start, b_end), attribute_dict=attr)

        res.append(new_text)

    return res


def __sample_diff_annotations(
    n_samples: int, con: sqlite3.Connection
) -> list[TextDiffRecord]:
    return (
        con.cursor()
        .execute(
            """
            SELECT t.text,
                   d.a_start, d.a_end, d.a_annt,
                   d.b_start, d.b_end, d.b_annt 
            FROM diff_annotations d
            JOIN texts t ON d.text_id = t.id
            ORDER BY RANDOM()
            LIMIT ?
            """,
            [n_samples],
        )
        .fetchall()
    )


def sample_diff_task(
    layer_a: str,
    layer_b: str,
    n_samples: int,
    label_attribute: str = "nertag",
    class_labels: list[str] = ["PER", "ORG", "LOC"],
    header: str = "Choose text with correct annotation",
    sqlite_db: str = "difftag.db",
    out_dir: str = "difftag",
):
    con = sqlite3.connect(sqlite_db)
    print(f"Conneted to sqlite databse: {sqlite_db}")

    samples = __sample_diff_annotations(n_samples, con)
    con.close()

    if len(samples) < n_samples:
        warnings.warn(
            f"Sampled less annotations than requested, got {len(samples)}, requested {n_samples}"
        )

    task_texts = __attach_diff_annotations(samples, layer_a, layer_b, label_attribute)

    conf = DiffTaggingConfiguration(
        layer_a=layer_a,
        layer_b=layer_b,
        class_labels=class_labels,
        header=header,
    )

    task = DiffTaggingTask(configuration=conf, label_attribute=label_attribute)

    os.makedirs(out_dir, exist_ok=True)
    task.export_data(task_texts, os.path.join(out_dir, "diff_tagging_task.json"))
    with open(os.path.join(out_dir, "diff_tagging_interface.xml"), "w") as f:
        f.write(task.interface_file)

    print(f"\nWrote label-studio task and interface to out dir: {out_dir}")


__all__ = ["sample_diff_task"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create label-studio diff tagging task by sampling from an sqlite database created by 'diff_collect.py' script"
    )

    parser.add_argument("--layer-a", "-a", type=str, help="Text layer a", required=True)
    parser.add_argument("--layer-b", "-b", type=str, help="Text layer b", required=True)

    parser.add_argument(
        "--label-attribute",
        "-l",
        type=str,
        help="Span annotation attribute, 'nertag' by default",
        default="nertag",
    )

    parser.add_argument(
        "--class-labels",
        type=str,
        nargs="+",
        help="List of class labels, 'PER ORG LOC' by default",
        default=["PER", "ORG", "LOC"],
    )

    parser.add_argument(
        "--header",
        type=str,
        help="Label studio task header, 'Choose text with correct annotation' by default",
        default="Choose text with correct annotation",
    )

    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        help="Number of samples",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="difftag",
        help="Out dir, 'difftag' by default",
    )

    parser.add_argument(
        "--sqlite-db",
        default="difftag.db",
        type=str,
        help="Input sqlite database file, 'difftag.db' by default",
    )

    args = parser.parse_args()

    sample_diff_task(
        layer_a=args.layer_a,
        layer_b=args.layer_b,
        label_attribute=args.label_attribute,
        class_labels=args.class_labels,
        header=args.header,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
    )
