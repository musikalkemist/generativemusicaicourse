"""
melody_generator.py

This script defines the MelodyGenerator class, which is responsible for generating
melodies using a trained Transformer model. The class offers functionality to produce
a sequence of musical notes, starting from a given seed sequence and extending it
to a specified maximum length.

The MelodyGenerator class leverages the trained Transformer model's ability to
predict subsequent notes in a melody based on the current sequence context. It
achieves this by iteratively appending each predicted note to the existing sequence
and feeding this extended sequence back into the model for further predictions.

This iterative process continues until the generated melody reaches the desired length
or an end-of-sequence token is predicted. The class utilizes a tokenizer to encode and
decode note sequences to and from the format expected by the Transformer model.

Key Components:
- MelodyGenerator: The primary class defined in this script, responsible for the
  generation of melodies.

Usage:
The MelodyGenerator class can be instantiated with a trained Transformer model
and an appropriate tokenizer. Once instantiated, it can generate melodies by
calling the `generate` method with a starting note sequence.

Note:
This class is intended to be used with a Transformer model that has been
specifically trained for melody generation tasks.
"""

import tensorflow as tf


class MelodyGenerator:
    """
    Class to generate melodies using a trained Transformer model.

    This class encapsulates the inference logic for generating melodies
    based on a starting sequence.
    """

    def __init__(self, transformer, tokenizer, max_length=50):
        """
        Initializes the MelodyGenerator.

        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, start_sequence):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        input_tensor = self._get_input_tensor(start_sequence)

        num_notes_to_generate = self.max_length - len(input_tensor[0])

        for _ in range(num_notes_to_generate):
            predictions = self.transformer(
                input_tensor, input_tensor, False, None, None, None
            )
            predicted_note = self._get_note_with_highest_score(predictions)
            input_tensor = self._append_predicted_note(
                input_tensor, predicted_note
            )

        generated_melody = self._decode_generated_sequence(input_tensor)

        return generated_melody

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the Transformer model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (tf.Tensor): The input tensor for the model.
        """
        input_sequence = self.tokenizer.texts_to_sequences([start_sequence])
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor

    def _get_note_with_highest_score(self, predictions):
        """
        Gets the note with the highest score from the predictions.

        Parameters:
            predictions (tf.Tensor): The predictions from the model.

        Returns:
            predicted_note (int): The index of the predicted note.
        """
        latest_predictions = predictions[:, -1, :]
        predicted_note_index = tf.argmax(latest_predictions, axis=1)
        predicted_note = predicted_note_index.numpy()[0]
        return predicted_note

    def _append_predicted_note(self, input_tensor, predicted_note):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            (tf.Tensor): The input tensor with the predicted note
        """
        return tf.concat([input_tensor, [[predicted_note]]], axis=-1)

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (tf.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_sequence_array = generated_sequence.numpy()
        generated_melody = self.tokenizer.sequences_to_texts(
            generated_sequence_array
        )[0]
        return generated_melody
