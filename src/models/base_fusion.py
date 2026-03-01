# src/models/base_fusion.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam


def build_base_model(config):

    # Inputs
    behavior_input = Input(shape=(2,), name="behavior_input")
    physio_input = Input(shape=(2,), name="physio_input")

    # Behavioral branch
    x1 = Dense(config.dense_units, activation='relu')(behavior_input)
    x1 = Dropout(config.dropout_rate)(x1)
    z_behavior = Dense(32, activation='relu')(x1)

    # Physiological branch
    x2 = Dense(config.dense_units, activation='relu')(physio_input)
    x2 = Dropout(config.dropout_rate)(x2)
    z_physio = Dense(32, activation='relu')(x2)

    # Simple Concatenation Fusion (NO ATTENTION)
    fusion = Concatenate()([z_behavior, z_physio])
    fusion = Dense(64, activation='relu')(fusion)
    fusion = Dropout(config.dropout_rate)(fusion)

    # Outputs
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
        loss_weights={
            'classification': 1.0,
            'hli': 0.2
        },
        metrics={
            'classification': 'accuracy',
            'hli': 'mse'
        }
    )

    return model
