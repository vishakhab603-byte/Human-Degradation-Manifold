class HDMConfig:

    # Data
    n_samples = 1000
    num_classes = 3

    # Model
    dense_units = 32
    dropout_rate = 0.3
    num_heads = 2
    learning_rate = 0.001

    # Training
    epochs = 20
    batch_size = 32

    # Flags
    use_attention = True
    use_uncertainty = True
