"""
melody_preprocessor.py

This script defines the MelodyPreprocessor class, a utility for preparing melody
datasets for training in a sequence-to-sequence Transformer model. The class
focuses on processing melody data by tokenizing and encoding the melodies, and
subsequently creating TensorFlow datasets suitable for training sequence-to-sequence
models.

The MelodyPreprocessor handles the entire preprocessing pipeline including loading
melodies from a dataset file, parsing the melodies into individual notes, tokenizing
and encoding these notes, and forming input-target pairs for model training. It
also includes functionality for padding sequences to a uniform length.

Key Features:
- Tokenization and encoding of melodies.
- Dynamic calculation of maximum sequence length based on the dataset.
- Creation of input-target pairs for sequence-to-sequence training.
- Conversion of processed data into TensorFlow datasets.

Usage:
To use the MelodyPreprocessor, initialize it with the path to a dataset containing
melodies and the desired batch size. Then call `create_training_dataset` to prepare
the dataset for training a Transformer model.


Note:
This script is intended to be used with datasets containing melody sequences in a
specific format, where each melody is represented as a string of comma-separated
musical notes (pitch with octave + duration in quarter length).
"""


import json

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer


class MelodyPreprocessor:
    """
    A class for preprocessing melodies for a Transformer model.

    This class takes melodies, tokenizes and encodes them, and prepares
    TensorFlow datasets for training sequence-to-sequence models.
    """

    def __init__(self, dataset_path, batch_size=32):
        """
        Initializes the MelodyPreprocessor.

        Parameters:
            dataset_path (str): Path to the dataset file.
            max_melody_length (int): Maximum length of the sequences.
            batch_size (int): Size of each batch in the dataset.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.tokenizer = Tokenizer(filters="", lower=False, split=",")
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        """
        Returns the number of tokens in the vocabulary including padding.

        Returns:
            int: The number of tokens in the vocabulary including padding.
        """
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        """
        Preprocesses the melody dataset and creates sequence-to-sequence
        training data.

        Returns:
            tf_training_dataset: A TensorFlow dataset containing input-target
                pairs suitable for training a sequence-to-sequence model.
        """
        dataset = self._load_dataset()
        parsed_melodies = [self._parse_melody(melody) for melody in dataset]
        tokenized_melodies = self._tokenize_and_encode_melodies(
            parsed_melodies
        )
        self._set_max_melody_length(tokenized_melodies)
        self._set_number_of_tokens()
        input_sequences, target_sequences = self._create_sequence_pairs(
            tokenized_melodies
        )
        tf_training_dataset = self._convert_to_tf_dataset(
            input_sequences, target_sequences
        )
        return tf_training_dataset

    def _load_dataset(self):
        """
        Loads the melody dataset from a JSON file.

        Returns:
            list: A list of melodies from the dataset.
        """
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def _parse_melody(self, melody_str):
        """
        Parses a single melody string into a list of notes.

        Parameters:
            melody_str (str): A string representation of a melody.

        Returns:
            list: A list of notes extracted from the melody string.
        """
        return melody_str.split(", ")

    def _tokenize_and_encode_melodies(self, melodies):
        """
        Tokenizes and encodes a list of melodies.

        Parameters:
            melodies (list): A list of melodies to be tokenized and encoded.

        Returns:
            tokenized_melodies: A list of tokenized and encoded melodies.
        """
        self.tokenizer.fit_on_texts(melodies)
        tokenized_melodies = self.tokenizer.texts_to_sequences(melodies)
        return tokenized_melodies

    def _set_max_melody_length(self, melodies):
        """
        Sets the maximum melody length based on the dataset.

        Parameters:
            melodies (list): A list of tokenized melodies.
        """
        self.max_melody_length = max([len(melody) for melody in melodies])

    def _set_number_of_tokens(self):
        """
        Sets the number of tokens based on the tokenizer.
        """
        self.number_of_tokens = len(self.tokenizer.word_index)

    def _create_sequence_pairs(self, melodies):
        """
        Creates input-target pairs from tokenized melodies.

        Parameters:
            melodies (list): A list of tokenized melodies.

        Returns:
            tuple: Two numpy arrays representing input sequences and target sequences.
        """
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1 : i + 1]  # Shifted by one time step
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)
        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum sequence length.

        Parameters:
            sequence (list): The sequence to be padded.

        Returns:
            list: The padded sequence.
        """
        return sequence + [0] * (self.max_melody_length - len(sequence))

    def _convert_to_tf_dataset(self, input_sequences, target_sequences):
        """
        Converts input and target sequences to a TensorFlow Dataset.

        Parameters:
            input_sequences (list): Input sequences for the model.
            target_sequences (list): Target sequences for the model.

        Returns:
            batched_dataset (tf.data.Dataset): A batched and shuffled
                TensorFlow Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, target_sequences)
        )
        shuffled_dataset = dataset.shuffle(buffer_size=1000)
        batched_dataset = shuffled_dataset.batch(self.batch_size)
        return batched_dataset


if __name__ == "__main__":
    # Usage example
    preprocessor = MelodyPreprocessor("dataset.json", batch_size=32)
    training_dataset = preprocessor.create_training_dataset()
