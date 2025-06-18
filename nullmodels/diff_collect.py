import os
import sqlite3
import warnings

from estnltk import Text
from estnltk.storage.postgres import PostgresStorage
from estnltk_core.layer_operations import diff_layer

type DiffRecord = tuple[
    int,
    int | None,
    int | None,
    str | None,
    int | None,
    int | None,
    str | None,
]


def __collect_diff_layer(
    text: Text, id: int, layer_a: str, layer_b: str, label_attr: str
) -> tuple[list[DiffRecord], bool]:
    """
    Collect conflicting annotations in shape of database records.
    """

    skip = False
    if layer_a not in text.layers:
        skip = True
        warnings.warn(f"Text {id} is missing layer '{layer_a}'")

    if layer_b not in text.layers:
        skip = True
        warnings.warn(f"Text {id} is missing layer '{layer_b}'")

    if skip:
        warnings.warn(f"Skipped text {id}")
        return [], True

    res = []
    for span_a, span_b in diff_layer(text[layer_a], text[layer_b]):
        a_start, a_end, a_annt = None, None, None
        b_start, b_end, b_annt = None, None, None

        if span_a:
            a_start, a_end = span_a.start, span_a.end
            a_annt = span_a.annotations[0][label_attr]

        if span_b:
            b_start, b_end = span_b.start, span_b.end
            b_annt = span_b.annotations[0][label_attr]

        record = (id, a_start, a_end, a_annt, b_start, b_end, b_annt)
        res.append(record)

    return res, False


def __insert_raw_text(id: int, text: Text, con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("INSERT INTO texts (id, text) VALUES (?, ?)", [id, text.text])
    con.commit()


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


def __setup_sqlite(con: sqlite3.Connection):
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


def collect_diff_annotations(
    layer_a: str,
    layer_b: str,
    label_attribute: str,
    pgpass_file: str,
    pg_dbname: str,
    pg_collection: str,
    pg_schema: str = "public",
    sqlite_db: str = "difftag.db",
):
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
    label_attr : str
        Span annotation attribute.
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
    con = sqlite3.connect(sqlite_db)
    print(f"Conneted to sqlite databse: {sqlite_db}")

    __setup_sqlite(con)

    storage = PostgresStorage(
        pgpass_file=pgpass_file,
        dbname=pg_dbname,
        schema=pg_schema,
    )

    collection = storage[pg_collection]

    stats = dict(n_texts=0, n_skipped=0, n_records=0)
    for id, text in collection.select(layers=[layer_a, layer_b]):  # type: ignore
        assert isinstance(id, int), "unreachable"
        assert isinstance(text, Text), "unreachable"

        diff_records, skipped = __collect_diff_layer(
            text, id, layer_a, layer_b, label_attribute
        )

        stats["n_texts"] += 1
        if len(diff_records) == 0 or skipped:
            stats["n_skipped"] += 1
            print(f"Found no conflicting annotations in text {id}")
            continue

        __insert_raw_text(id, text, con)
        __insert_diff_layer(diff_records, con)

    storage.close()
    con.close()

    print(
        f"\nTexts:\t{stats['n_texts']}\nSkipped:\t{stats['n_skipped']}\nAnnotations:\t{stats['n_records']}"
    )


__all__ = ["collect_diff_annotations"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect conflicting span annotations between two layers of texts in a postgres collection into an sqlite database"
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
        "--pgpass-file",
        type=str,
        help="Postgres password file",
        required=True,
    )

    parser.add_argument(
        "--pg-dbname",
        type=str,
        help="Postgres database name",
        required=True,
    )

    parser.add_argument(
        "--pg-schema",
        type=str,
        help="Postgres schema, 'public' by default",
        default="public",
    )

    parser.add_argument(
        "--pg-collection",
        type=str,
        help="Postgres collection",
        required=True,
    )

    parser.add_argument(
        "--sqlite-db",
        default="difftag.db",
        type=str,
        help="Output  sqlite database file, 'difftag.db' by default",
    )

    parser.add_argument(
        "--drop-sqlite-db",
        action="store_true",
        help="Dry run, if present delete sqlite database after running",
    )

    args = parser.parse_args()

    collect_diff_annotations(
        layer_a=args.layer_a,
        layer_b=args.layer_b,
        label_attribute=args.label_attribute,
        pgpass_file=args.pgpass_file,
        pg_dbname=args.pg_dbname,
        pg_schema=args.pg_schema,
        pg_collection=args.pg_collection,
        sqlite_db=args.sqlite_db,
    )

    if args.drop_sqlite_db:
        os.remove(args.sqlite_db)
        print(f"\nDeleted sqlite database: {args.sqlite_db}")
    else:
        print(f"\nSaved to sqlite database: {args.sqlite_db}")
