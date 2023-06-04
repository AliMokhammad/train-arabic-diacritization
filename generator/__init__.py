import tensorflow as tf
from tensorflow.keras import layers

class Generator():
    def __init__(self,
                vocab_size=1000, 
                max_sequence_length=100, 
                embedding_dim=256, 
                num_transformer_layers=4, 
                num_heads=10, 
                hidden_dim=512, 
                num_diacritics=14
        ):
        inputs = layers.Input(shape=(max_sequence_length,), dtype=tf.int32)

        # Embedding layer
        embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

        # Transformer-based Encoder Layers
        x = embedding_layer

        for _ in range(num_transformer_layers):
            x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Dense(units=hidden_dim, activation="relu")(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Transformer-based Decoder Layers
        for _ in range(num_transformer_layers):
            x = layers.Attention(use_scale=True)([x, x])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Dense(units=hidden_dim, activation="relu")(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Output Layer
        outputs = layers.Dense(units=num_diacritics, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Generator")
        return model
