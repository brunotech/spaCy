#!/usr/bin/env python
# coding: utf8
"""Visualize spaCy word vectors in Tensorboard.

Adapted from: https://gist.github.com/BrikerMan/7bd4e4bd0a00ac9076986148afc06507
"""
from __future__ import unicode_literals

from os import path

import tqdm
import math
import numpy
import plac
import spacy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins.projector import (
    visualize_embeddings,
    ProjectorConfig,
)


@plac.annotations(
    vectors_loc=("Path to spaCy model that contains vectors", "positional", None, str),
    out_loc=(
        "Path to output folder for tensorboard session data",
        "positional",
        None,
        str,
    ),
    name=(
        "Human readable name for tsv file and vectors tensor",
        "positional",
        None,
        str,
    ),
)
def main(vectors_loc, out_loc, name="spaCy_vectors"):
    meta_file = f"{name}.tsv"
    out_meta_file = path.join(out_loc, meta_file)

    print(f"Loading spaCy vectors model: {vectors_loc}")
    model = spacy.load(vectors_loc)
    print(f"Finding lexemes with vectors attached: {vectors_loc}")
    strings_stream = tqdm.tqdm(
        model.vocab.strings, total=len(model.vocab.strings), leave=False
    )
    queries = [w for w in strings_stream if model.vocab.has_vector(w)]
    vector_count = len(queries)

    print(
        f"Building Tensorboard Projector metadata for ({vector_count}) vectors: {out_meta_file}"
    )

    # Store vector data in a tensorflow variable
    tf_vectors_variable = numpy.zeros((vector_count, model.vocab.vectors.shape[1]))

    # Write a tab-separated file that contains information about the vectors for visualization
    #
    # Reference: https://www.tensorflow.org/programmers_guide/embedding#metadata
    with open(out_meta_file, "wb") as file_metadata:
        # Define columns in the first row
        file_metadata.write("Text\tFrequency\n".encode("utf-8"))
        for vec_index, text in enumerate(tqdm.tqdm(queries, total=len(queries), leave=False)):
            # https://github.com/tensorflow/tensorflow/issues/9094
            text = "<Space>" if text.lstrip() == "" else text
            lex = model.vocab[text]

            # Store vector data and metadata
            tf_vectors_variable[vec_index] = model.vocab.get_vector(text)
            file_metadata.write(
                f"{text}\t{math.exp(lex.prob) * vector_count}\n".encode(
                    "utf-8"
                )
            )
    print("Running Tensorflow Session...")
    sess = tf.InteractiveSession()
    tf.Variable(tf_vectors_variable, trainable=False, name=name)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(out_loc, sess.graph)

    # Link the embeddings into the config
    config = ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = name
    embed.metadata_path = meta_file

    # Tell the projector about the configured embeddings and metadata file
    visualize_embeddings(writer, config)

    # Save session and print run command to the output
    print("Saving Tensorboard Session...")
    saver.save(sess, path.join(out_loc, f"{name}.ckpt"))
    print("Done. Run `tensorboard --logdir={0}` to view in Tensorboard".format(out_loc))


if __name__ == "__main__":
    plac.call(main)
