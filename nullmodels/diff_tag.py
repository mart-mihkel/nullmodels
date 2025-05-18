import json
import sqlite3
import warnings
import numpy as np

from estnltk import Layer, Span, Text
from estnltk_core.layer_operations import diff_layer

from typing import Tuple, List, Union

from estnltk.converters.label_studio.labelling_tasks import PhraseTaggingTask

type DiffRecord = Tuple[
    int,
    str,
    Union[int, None],
    Union[int, None],
    Union[str, None],
    Union[int, None],
    Union[int, None],
    Union[str, None],
]

type TextDiffRecord = Tuple[
    str,
    Union[int, None],
    Union[int, None],
    Union[str, None],
    Union[int, None],
    Union[int, None],
    Union[str, None],
]


def import_labelstudio(tasks_json: str) -> List[Text]:
    """
    ...
    """
    tasks = json.loads(tasks_json)
    texts = []
    for task in tasks:
        data = task["data"]
        text = Text(data["text"])
        layer = Layer(name="validated_nertag", ambiguous=True)

        attributes = {}
        for annotation in task["annotations"]:
            for span in annotation["result"]:
                value = span["value"]
                start = value["start"]
                end = value["end"]

                labels = value["labels"][0]
                attributes["nertag"] = labels
                layer.add_annotation((start, end), attribute_dict=attributes)

        text.add_layer(layer)
        texts.append(text)

    return texts


def export_labelstudio(n_samp: int) -> Tuple[str, str]:
    """
    Sample random conflicting nertag annotations, return label studio tasks
    and interfrace

    Parameters
    ----------
    n_samp : int
        Number of annotations to sample

    Returns
    -------
    (interface, tasks) : Tuple[str, str]
        Label studio interface xml and tasks json
    """

    iface = """
    <View>
        <Header value="Choose correct named entity annotation" />
        <Choices name="validated_nertag" toName="text_a" choice="single-radio" showInline="true" >
            <Choice value="A" />
            <Choice value="B" />
            <Choice value="None" />
        </Choices>
        <Labels name="ner_labels" toName="text_a" visible="false" >
            <Label value="PER" background="green" hotkey="none" />
            <Label value="ORG" background="orange" hotkey="none" />
            <Label value="LOC" background="blue" hotkey="none" />
        </Labels>
        <Header value="Annotation A" />
        <Text name="text_a" value="$text_a" granularity="word" showLabels="true" />
        <Header value="Annotation B" />
        <Text name="text_b" value="$text_b" granularity="word" showLabels="true" />
    </View>
    """

    tasks = __sample_label_studio_tasks(n_samp)

    return iface, tasks


def __sample_label_studio_tasks(n_samp: int) -> str:
    records = __sample_diff_annotations(n_samp)
    n = len(records)
    if n < n_samp:
        warnings.warn(
            f"There are less conflicting annotations than requested, got {n}, requested {n_samp}"
        )

    tasks = [dict()] * n
    for i, record in enumerate(records):
        text, a_start, a_end, a_annt, b_start, b_end, b_annt = record
        results = []

        if a_start is not None and a_end and a_annt:
            res = {
                "from_name": "ner_labels",
                "to_name": "text_a",
                "type": "labels",
                "value": {"start": a_start, "end": a_end, "labels": [a_annt]},
            }

            results.append(res)

        if b_start is not None and b_end and b_annt:
            res = {
                "from_name": "ner_labels",
                "to_name": "text_b",
                "type": "labels",
                "value": {"start": b_start, "end": b_end, "labels": [b_annt]},
            }

            results.append(res)

        tasks[i] = {
            "data": {"text_a": text, "text_b": text},
            "annotations": [{"result": results}],
        }

    return json.dumps(tasks)


def persist_diff_layer(texts: List[Text], layer_a: str, layer_b: str):
    """
    Find and save records of different annotations between layers in an sqlite
    database

    Parameters
    ----------
    texts : List[Text]
        Texts to process
    layer_a : str
        First layer to compare, must be present in all texts
    layer_b : str
        Second layer to compare, must be present in all texts
    """
    texts_ids = [(i, t.text) for i, t in enumerate(texts)]
    diffs = [__collect_layer_diff(t, i, layer_a, layer_b) for i, t in enumerate(texts)]
    records = sum(diffs, start=[])

    __setup_db()
    __persist_raw_texts(texts_ids)
    __persist_diff_records(records)


def __collect_layer_diff(
    text: Text, text_id: int, layer_a: str, layer_b: str
) -> List[DiffRecord]:
    """
    Collect records of different annotations between layers

    Parameters
    ----------
    text : Text
        Text object with layers `layer_a` and `layer_b`
    text_id : int
        Text id  # TODO: this is sus
    layer_a : str
        First layer to compare, must be present in `text`
    layer_b : str
        Second layer to compare, must be present in `text`

    Returns
    -------
    diff_records : List[DiffRecord]
        Records of different annotations between layers
    """
    assert layer_a in text.layers, f"Text is missing '{layer_a}' layer"
    assert layer_b in text.layers, f"Text is missing '{layer_b}' layer"

    records = []
    for span_a, span_b in diff_layer(text[layer_a], text[layer_b]):
        status = "conflict"

        if span_a is None:
            a_start, a_end, a_annt = None, None, None
            status = "missing"
        else:
            a_start, a_end = span_a.start, span_a.end
            a_annt = __get_nertag_annt(span_a)

        if span_b is None:
            b_start, b_end, b_annt = None, None, None
            status = "extra"
        else:
            b_start, b_end = span_b.start, span_b.end
            b_annt = __get_nertag_annt(span_b)

        record = (text_id, status, a_start, a_end, a_annt, b_start, b_end, b_annt)
        records.append(record)

    return records


def __get_nertag_annt(span: Span) -> str:
    annt = map(dict, span.annotations)
    annt = map(lambda a: a.get("nertag"), annt)
    annt = filter(lambda a: a is not None, annt)
    annt = next(annt)

    if annt is None:
        raise RuntimeError(f"Encountered span with no nertag annotation: {span}")

    return annt


def __sample_diff_annotations(
    n_samp: int, db_file="diff_tag.db"
) -> List[TextDiffRecord]:
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    records = cur.execute(
        """
        SELECT t.text,
               d.a_start, d.a_end, d.a_annt,
               d.b_start, d.b_end, d.b_annt 
        FROM diff_annotations d
        JOIN texts t ON d.text_id = t.id
        ORDER BY RANDOM()
        LIMIT ?
        """,
        [n_samp],
    ).fetchall()
    con.close()

    return records


def __persist_raw_texts(texts: List[Tuple[int, str]], db_file="diff_tag.db"):
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.executemany("INSERT INTO texts (id, text) VALUES (?, ?)", texts)
    con.commit()
    con.close()


def __persist_diff_records(records: List[DiffRecord], db_file="diff_tag.db"):
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO diff_annotations  (
            text_id, status, 
            a_start, a_end, a_annt, 
            b_start, b_end, b_annt
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    con.commit()
    con.close()


def __setup_db(db_file="diff_tag.db"):
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS diff_annotations (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            text_id INT,
            status  TEXT CHECK (status IN ('extra', 'missing', 'conflict')),
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
            id      INTEGER PRIMARY KEY,
            text    TEXT
        )
    """)

    con.close()


__all__ = [
    "export_labelstudio",
    "persist_diff_layer",
]
