
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate,
    LayerNormalization, MultiHeadAttention, Lambda
)
from tensorflow.keras.optimizers import Adam


def build_attention_model(config):

    behavior_input = Input(shape=(2,), name="behavior_input")
    physio_input = Input(shape=(2,), name="physio_input")

    x1 = Dense(config.dense_units, activation='relu')(behavior_input)
    x1 = Dropout(config.dropout_rate)(x1)
    z_behavior = Dense(32, activation='relu')(x1)

    x2 = Dense(config.dense_units, activation='relu')(physio_input)
    x2 = Dropout(config.dropout_rate)(x2)
    z_physio = Dense(32, activation='relu')(x2)

    z_behavior_exp = Lambda(lambda x: tf.expand_dims(x, axis=1))(z_behavior)
    z_physio_exp = Lambda(lambda x: tf.expand_dims(x, axis=1))(z_physio)

    attention = MultiHeadAttention(
        num_heads=config.num_heads,
        key_dim=16
    )(
        query=z_behavior_exp,
        value=z_physio_exp,
        key=z_physio_exp
    )

    attention = Lambda(lambda x: x[0] + x[1])(
        [attention, z_behavior_exp]
    )
    attention = LayerNormalization()(attention)
    attention = Lambda(lambda x: tf.squeeze(x, axis=1))(attention)

    fusion = Concatenate()([z_behavior, z_physio, attention])
    fusion = Dense(64, activation='relu')(fusion)
    fusion = Dropout(config.dropout_rate)(fusion)

    class_output = Dense(
        config.num_classes,
        activation='softmax',
        name='classification'
    )(fusion)

    hli_output = Dense(
        1,
        activation='linear',
        name='hli'
    )(fusion)

    model = Model(
        inputs=[behavior_input, physio_input],
        outputs=[class_output, hli_output]
    )

    model.compile(
        optimizer=Adam(config.learning_rate),
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
