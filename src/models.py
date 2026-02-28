# src/models.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

def build_fusion_model(
    input_dim_behavior=2,
    input_dim_physio=2,
    dense_units=32,
    dropout_rate=0.3,
    learning_rate=0.001
):

    # Behavioral Branch
    behavior_input = Input(shape=(input_dim_behavior,), name="behavior_input")
    x1 = Dense(dense_units, activation='relu')(behavior_input)
    x1 = Dropout(dropout_rate)(x1)
    z_behavior = Dense(16, activation='relu')(x1)

    # Physiological Branch
    physio_input = Input(shape=(input_dim_physio,), name="physio_input")
    x2 = Dense(dense_units, activation='relu')(physio_input)
    x2 = Dropout(dropout_rate)(x2)
    z_physio = Dense(16, activation='relu')(x2)

    # Fusion Layer
    fusion = Concatenate()([z_behavior, z_physio])
    fusion = Dense(32, activation='relu')(fusion)

    # Classification Output
    class_output = Dense(3, activation='softmax', name='classification')(fusion)

    # Regression Output (Human Limit Index)
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
