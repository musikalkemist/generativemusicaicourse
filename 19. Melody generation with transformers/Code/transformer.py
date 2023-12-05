"""
This script defines the Transformer model, a state-of-the-art model architecture used
for a variety of natural language processing tasks, adapted here for music generation.
The Transformer model relies on the mechanism of attention, differentially weighting
the significance of different input elements.

The script includes the implementation of various components of the Transformer
architecture, including the Encoder, Decoder, and their respective layers. It also
defines a sinusoidal positional encoding function that provides the model with
information about the relative position of tokens in the sequence.

Key Components:
- Transformer: The main model class combining the Encoder and Decoder.
- Encoder: Processes the input sequence and generates a context-rich representation.
- Decoder: Generates the output sequence based on the Encoder's output and its own
  input.
- EncoderLayer and DecoderLayer: Individual layers used in the Encoder and Decoder.
- _get_angles and sinusoidal_position_encoding: Functions to generate positional
  encoding based on the sequence length and model dimensionality.

Usage:
To use the Transformer model, instantiate it with the required dimensions, number
of layers, vocabulary sizes, and other parameters. The model can then be used for
training or inference tasks in music generation or other sequence-to-sequence
transformations.

Note:
This implementation of the Transformer model is designed for flexibility and can be
adapted for various sequence-to-sequence tasks beyond music generation.
"""

import numpy as np
import tensorflow as tf
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
)


def sinusoidal_position_encoding(num_positions, d_model):
    """
    Compute positional encoding for a given position and dimension.

    Parameters:
        num_positions (int): Number of positions.
        d_model (int): Dimension of the model.

    Returns:
        Tensor: Positional encoding for the given position and dimension.
    """

    angles = _get_angles(
        np.arange(num_positions)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )

    # Apply sin to even indices in the array; 2i
    sines = np.sin(angles[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angles[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]  # (1, position, d_model)

    return tf.cast(pos_encoding, dtype=tf.float32)


def _get_angles(pos, i, d_model):
    """
    Compute the angles for the positional encoding.

    Parameters:
        pos (np.ndarray): Positions.
        i (np.ndarray): Indices.
        d_model (int): Dimension of the model.

    Returns:
        np.ndarray: Angles for the positional encoding.
    """
    angle_dropout_rates = 1 / np.power(
        10000, (2 * (i // 2)) / np.float32(d_model)
    )
    return pos * angle_dropout_rates


class Transformer(tf.keras.Model):
    """
    The Transformer model architecture, consisting of an Encoder and Decoder.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        max_num_positions_in_pe_encoder,
        max_num_positions_in_pe_decoder,
        dropout_rate=0.1,
    ):
        """
        Parameters:
            num_layers (int): Number of layers in both Encoder and Decoder.
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_feedforward (int): Dimension of the feed forward network.
            input_vocab_size (int): Size of the input vocabulary.
            target_vocab_size (int): Size of the target vocabulary.
            max_num_positions_in_pe_encoder (int): The maximum positions for input.
            max_num_positions_in_pe_decoder (int): The maximum positions for
                target.
            dropout_rate (float): Dropout dropout_rate.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            d_feedforward,
            input_vocab_size,
            max_num_positions_in_pe_encoder,
            dropout_rate,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            d_feedforward,
            target_vocab_size,
            max_num_positions_in_pe_decoder,
            dropout_rate,
        )

        self.final_layer = Dense(target_vocab_size)

    def call(
        self,
        input,
        target,
        training,
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask,
    ):
        """
        Process the input through the Transformer model.

        Parameters:
            input (Tensor): Input tensor to the Encoder.
            target (Tensor): Target tensor for the Decoder.
            training (bool): Whether the layer should behave in training mode.
            enc_padding_mask (Tensor): Padding mask for the Encoder.
            look_ahead_mask (Tensor): Look-ahead mask for the Decoder.
            dec_padding_mask (Tensor): Padding mask for the Decoder.

        Returns:
            Tensor: The final output of the Transformer.
            dict: Attention weights from the Decoder layers.
        """
        enc_output = self.encoder(
            input, training, enc_padding_mask
        )  # (batch_size, input_seq_len, d_model)

        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, dec_padding_mask
        )  # (batch_size, tar_seq_len, d_model)

        logits = self.final_layer(
            dec_output
        )  # (batch_size, target_seq_len, target_vocab_size)

        return logits


class Encoder(tf.keras.layers.Layer):
    """
    The Encoder of a Transformer model, consisting of multiple EncoderLayers.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        maximum_positions_in_pe,
        dropout_rate=0.1,
    ):
        """
        Parameters
            num_layers (int): Number of EncoderLayers.
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_feedforward (int): Dimension of the feed forward network.
            input_vocab_size (int): Size of the input vocabulary.
            maximum_positions_in_pe (int): The maximum sequence length that
                this model might ever be used with.
            dropout_rate (float): Dropout dropout_rate.
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(
            maximum_positions_in_pe, d_model
        )
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Process the input through the Encoder.

        Args:
            x (Tensor): Input tensor.
            training (bool): Whether the layer should behave in training mode.
            mask (Tensor): Mask to be applied on attention weights.

        Returns:
            Tensor: Output of the Encoder.
        """
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        sliced_pos_encoding = self._get_sliced_positional_encoding(x)
        x += sliced_pos_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def _get_sliced_positional_encoding(self, x):
        """
        Get a slice of the full positional encoding.

        Patameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: A slice of the full positional encoding.
        """
        number_of_tokens = x.shape[1]
        return self.pos_encoding[:, :number_of_tokens, :]


class Decoder(tf.keras.layers.Layer):
    """
    The Decoder of a Transformer model, consisting of multiple DecoderLayers.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        target_vocab_size,
        maximum_positions_in_pe,
        dropout_rate=0.1,
    ):
        """
        Parameters:
            num_layers (int): Number of DecoderLayers.
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_feedforward (int): Dimension of the feed forward network.
            target_vocab_size (int): Size of the target vocabulary.
            maximum_positions_in_pe (int): The maximum sequence length that
                this model might ever be used with.
            dropout_rate (float): Dropout dropout_rate.
        """
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(
            maximum_positions_in_pe, d_model
        )

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, d_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Process the input through the Decoder.

        Parameters:
            x (Tensor): Input tensor to the Decoder.
            enc_output (Tensor): Output from the Encoder.
            training (bool): Whether the layer should behave in training mode.
            look_ahead_mask (Tensor): Mask for the first MultiHeadAttention layer.
            padding_mask (Tensor): Mask for the second MultiHeadAttention layer.

        Returns:
            Tensor: The output of the Decoder.
        """

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        sliced_pos_encoding = self._get_sliced_positional_encoding(x)
        x += sliced_pos_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

        return x

    def _get_sliced_positional_encoding(self, x):
        """
        Get a slice of the full positional encoding.

        Patameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: A slice of the full positional encoding.
        """
        number_of_tokens = x.shape[1]
        return self.pos_encoding[:, :number_of_tokens, :]


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder Layer of a Transformer, consisting of MultiHeadAttention and
    Feed Forward Neural Network.
    """

    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        """
        Parameters:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_feedforward (int): Dimension of the feed forward network.
            dropout_rate (float): Dropout dropout_rate.
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(d_feedforward, activation="relu"), Dense(d_model)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Process the input through the Encoder layer.

        Parameters:
            x (Tensor): Input tensor.
            training (bool): Whether the layer should behave in training mode.
            mask (Tensor): Mask to be applied on attention weights.

        Returns:
            Tensor: Output of the Encoder layer.
        """
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder Layer of a Transformer, consisting of two MultiHeadAttention
    layers and a Feed Forward Neural Network.
    """

    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        """
        Parameters:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_feedforward (int): Dimension of the feed forward network.
            dropout_rate (float): Dropout dropout_rate.
        """
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)

        self.ffn = tf.keras.Sequential(
            [Dense(d_feedforward, activation="relu"), Dense(d_model)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Process the input through the Decoder layer.

        Parameters
            x (Tensor): Input tensor to the Decoder layer.
            enc_output (Tensor): Output from the Encoder.
            training (bool): Whether the layer should behave in training mode.
            look_ahead_mask (Tensor): Mask for the first MultiHeadAttention layer.
            padding_mask (Tensor): Mask for the second MultiHeadAttention layer.

        Returns:
            Tensor: The output of the Decoder layer.
        """
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(
            out1, enc_output, enc_output, attention_mask=padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


if __name__ == "__main__":
    # Define Transformer parameters
    num_layers = 2
    d_model = 64
    num_heads = 2
    d_feedforward = 128
    input_vocab_size = 100
    target_vocab_size = 100
    dropout_dropout_rate = 0.1
    pe_input = 10
    pe_target = 10

    # Instantiate the Transformer model
    transformer_model = Transformer(
        num_layers,
        d_model,
        num_heads,
        d_feedforward,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        dropout_dropout_rate,
    )

    # Dummy input shapes for encoder and decoder
    dummy_inp = tf.random.uniform(
        (1, 10), dtype=tf.int64, minval=0, maxval=input_vocab_size
    )
    dummy_tar = tf.random.uniform(
        (1, 10), dtype=tf.int64, minval=0, maxval=target_vocab_size
    )

    # Build the model using dummy input
    transformer_model(
        dummy_inp,
        dummy_tar,
        training=False,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    )

    # Display the model summary
    transformer_model.summary()
