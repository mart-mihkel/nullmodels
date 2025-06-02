import sqlite3
import warnings
import tempfile

from estnltk import Layer, Text
from estnltk_core.layer_operations import diff_layer

from collections import defaultdict

type DiffRecord = tuple[
    int,
    int | None,
    int | None,
    str | None,
    int | None,
    int | None,
    str | None,
]

type TextDiffRecord = tuple[
    str,
    int | None,
    int | None,
    str | None,
    int | None,
    int | None,
    str | None,
]


def __collect_diff_layer(
    texts: list[Text], text_ids: list[int], layer_a: str, layer_b: str
) -> list[DiffRecord]:
    """
    Collect conflicting annotations in shape of database records.
    """
    assert len(texts) == len(text_ids), "texts and text_ids have different shape"

    res = []
    for text, id in zip(texts, text_ids):
        assert layer_a in text.layers, f"Text {id} is missing '{layer_a}' layer"
        assert layer_b in text.layers, f"Text {id} is missing '{layer_b}' layer"

        for span_a, span_b in diff_layer(text[layer_a], text[layer_b]):
            a_start, a_end, a_annt = None, None, None
            b_start, b_end, b_annt = None, None, None

            if span_a:
                a_start, a_end = span_a.start, span_a.end
                a_annt = span_a.annotations[0]["nertag"]

            if span_b:
                b_start, b_end = span_b.start, span_b.end
                b_annt = span_b.annotations[0]["nertag"]

            record = (id, a_start, a_end, a_annt, b_start, b_end, b_annt)
            res.append(record)

    return res


def __attach_diff_annotations(
    texts: list[Text], layer_a: str, layer_b: str, annotations: defaultdict[str, list]
) -> list[Text]:
    """
    Create new texts with just the sampled annotations.
    """
    res = []
    for text in texts:
        new_text = Text(text.text)

        new_a = Layer(layer_a, attributes=["nertag"], ambiguous=True)
        new_b = Layer(layer_b, attributes=["nertag"], ambiguous=True)

        new_text.add_layer(new_a)
        new_text.add_layer(new_b)

        for annt in annotations[text.text]:
            a_start, a_end, a_annt, b_start, b_end, b_annt = annt

            if a_start is not None and a_end is not None and a_annt:
                attr = {"nertag": a_annt}
                new_a.add_annotation((a_start, a_end), attribute_dict=attr)

            if b_start is not None and b_end is not None and b_annt:
                attr = {"nertag": b_annt}
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


def __insert_raw_texts(texts: list[Text], con: sqlite3.Connection) -> list[int]:
    cur = con.cursor()

    ids = []
    for t in texts:
        cur.execute("INSERT INTO texts (text) VALUES (?)", [t.text])
        ids.append(cur.lastrowid)

    con.commit()

    return ids


def __insert_diff_layer(records: list[DiffRecord], con: sqlite3.Connection):
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO diff_annotations  (
            text_id,
            a_start, a_end, a_annt, 
            b_start, b_end, b_annt
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    con.commit()


def __setup_db(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS diff_annotations (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            text_id INT,
            a_start INT,
            a_end   INT,
            a_annt  TEXT,
            b_start INT,
            b_end   INT,
            b_annt  TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS texts (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            text    TEXT
        )
    """)


def sample_diff_annotations(
    texts: Text | list[Text],
    layer_a: str,
    layer_b: str,
    n_samples: int,
    sqlite_db_path: str | None = None,
    verbose: bool = False,
) -> list[Text]:
    """
    Collects all the conflicting, extra or missing nertag annotations in all
    the texts between layers a and b. Saves them in an sqlite databse. And
    uniformly samples from these and returns a list of new texts with only
    the sampled conflicting annotations.

    Parameters
    ----------
    texts : Text | list[Text]
        Texts to sample from.
    layer_a : str
        First layer to compare.
    layer_b : str
        Second layer to compare.
    n_samples : int
        Number of conflicting annotations to sample.
    sqlite_db_path : str | None, default '/tmp/diff_tag.db'
        Sqlite database file path.
    verbose : bool, default False
        Verbose

    Returns
    -------
    texts : list[Text]
        New texts with only the sampled annotations.
    """
    if isinstance(texts, Text):
        texts = [texts]

    if sqlite_db_path is None:
        sqlite_db_path = os.path.join(tempfile.gettempdir(), "diff_tag.db")

    con = sqlite3.connect(sqlite_db_path)
    if verbose:
        print(f"Created sqlite databse {sqlite_db_path}")

    __setup_db(con)
    text_ids = __insert_raw_texts(texts, con)
    diff_records = __collect_diff_layer(texts, text_ids, layer_a, layer_b)
    __insert_diff_layer(diff_records, con)

    samples = __sample_diff_annotations(n_samples, con)
    samples_grouped = defaultdict(list)
    for sample in samples:
        text, annt = sample[0], sample[1:]
        samples_grouped[text].append(annt)

    con.close()

    if len(samples) < n_samples:
        warnings.warn(
            "Sampled less annotations than requested, "
            f"got {len(samples)}, requested {n_samples}",
            UserWarning,
        )
    elif verbose:
        print(f"Sampled {len(samples)} conflicting annotations from {len(texts)} texts")

    texts_out = __attach_diff_annotations(texts, layer_a, layer_b, samples_grouped)

    return texts_out


__all__ = ["sample_diff_annotations"]


if __name__ == "__main__":
    import os
    import argparse

    from estnltk.converters import json_to_text, text_to_json

    parser = argparse.ArgumentParser(
        description="Uniformly sample conflicting NER "
        "annotations between two layers of multiple texts."
    )

    parser.add_argument(
        "--layer-a", "-a", type=str, help="First layer to compare", required=True
    )

    parser.add_argument(
        "--layer-b", "-b", type=str, help="Second layer to compare", required=True
    )

    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        help="Number of conflicting annotations to sample",
        required=True,
    )

    parser.add_argument(
        "--text-dir",
        "-d",
        type=str,
        help="Path to directory with json serialized texts",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        "-o",
        default="diff_tag_out",
        type=str,
        help="Directory to output texts with sampled conflicting annotations",
    )

    parser.add_argument(
        "--sqlite-db",
        default=os.path.join(tempfile.gettempdir(), "diff_tag.db"),
        type=str,
        help="Sqlite database file used for diff annotation sampling",
    )

    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="If present doesn't delete sqlite database after running script",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.text_dir):
        parser.error(f"{args.text_dir} is not a valid directory")

    if not os.path.isdir(args.out_dir) and os.path.exists(args.out_dir):
        parser.error(f"{args.out_dir} is not a valid directory")

    in_texts: list[Text] = []
    in_txt_files = os.listdir(args.text_dir)
    for in_txt_file in in_txt_files:
        in_txt_path = os.path.join(args.text_dir, in_txt_file)
        in_texts.append(json_to_text(file=in_txt_path))

    if args.verbose:
        print(f"Loaded {len(in_texts)} texts")

    out_texts = sample_diff_annotations(
        in_texts,
        args.layer_a,
        args.layer_b,
        n_samples=args.num_samples,
        sqlite_db_path=args.sqlite_db,
        verbose=args.verbose,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for out_text, out_txt_file in zip(out_texts, in_txt_files):
        out_txt_path = os.path.join(args.out_dir, out_txt_file)
        text_to_json(out_text, file=out_txt_path)

    if args.verbose:
        print(f"Wrote {len(out_texts)} to {args.out_dir}")

    if not args.keep_db:
        os.remove(args.sqlite_db)
        if args.verbose:
            print(f"Deleted sqlite3 database {args.sqlite_db}")
