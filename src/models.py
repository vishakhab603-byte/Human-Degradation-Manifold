# src/models.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate,
    LayerNormalization, MultiHeadAttention, Lambda
)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def build_fusion_model(
    input_dim_behavior=2,
    input_dim_physio=2,
    dense_units=32,
    dropout_rate=0.3,
    learning_rate=0.001,
    num_heads=2
):

    # Behavioral Branch
    behavior_input = Input(shape=(input_dim_behavior,), name="behavior_input")
    x1 = Dense(dense_units, activation='relu')(behavior_input)
    x1 = Dropout(dropout_rate)(x1)
    z_behavior = Dense(32, activation='relu')(x1)

    # Physiological Branch
    physio_input = Input(shape=(input_dim_physio,), name="physio_input")
    x2 = Dense(dense_units, activation='relu')(physio_input)
    x2 = Dropout(dropout_rate)(x2)
    z_physio = Dense(32, activation='relu')(x2)

    # Expand dims using Lambda
    z_behavior_exp = Lambda(lambda x: tf.expand_dims(x, axis=1))(z_behavior)
    z_physio_exp = Lambda(lambda x: tf.expand_dims(x, axis=1))(z_physio)

    # Cross-modal Attention
    attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=16)
    attention_output = attention_layer(
        query=z_behavior_exp,
        value=z_physio_exp,
        key=z_physio_exp
    )

    # Residual connection
    attention_output = Lambda(lambda x: x[0] + x[1])(
        [attention_output, z_behavior_exp]
    )
    attention_output = LayerNormalization()(attention_output)

    # Remove sequence dimension
    attention_output = Lambda(lambda x: tf.squeeze(x, axis=1))(attention_output)

    # Fusion
    fusion = Concatenate()([z_behavior, z_physio, attention_output])
    fusion = Dense(64, activation='relu')(fusion)
    fusion = Dropout(dropout_rate)(fusion)

    # Classification Output
    class_output = Dense(3, activation='softmax', name='classification')(fusion)

    # Regression Output
    hli_output = Dense(1, activation='linear', name='hli')(fusion)

    model = Model(
        inputs=[behavior_input, physio_input],
        outputs=[class_output, hli_output]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'classification': 'categorical_crossentropy',
            'hli': 'mse'
        },
        metrics={
            'classification': 'accuracy',
            'hli': 'mse'
        }
    )

    return model
