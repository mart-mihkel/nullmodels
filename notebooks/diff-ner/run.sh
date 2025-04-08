#!/usr/bin/env bash

schema="ner_diff"
collection="ner_diff_sampling"

old_layer="ner"
new_layer="webner"

python diff.py \
    --pgpass pgpass-dev \
    --schema $schema \
    --new_ner_layer $new_layer \
    $collection \
    $old_layer
