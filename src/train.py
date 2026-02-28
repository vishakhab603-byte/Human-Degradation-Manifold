# src/train.py

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Extract embeddings
latent_model = model.get_layer("fusion_dense")  # or correct layer name
latent_output = latent_model.output

# Build embedding extractor
from tensorflow.keras.models import Model
embedding_model = Model(inputs=model.input, outputs=latent_output)

embeddings = embedding_model.predict([Xb_test, Xp_test])

pca = PCA(n_components=2)
Z = pca.fit_transform(embeddings)

plt.scatter(Z[:,0], Z[:,1], c=y_test)
plt.title("HDM Latent Space")
plt.show()

from sklearn.metrics import confusion_matrix, roc_auc_score

y_pred = model.predict([Xb_test, Xp_test])[0]
y_pred_labels = y_pred.argmax(axis=1)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred_labels)
print("Confusion Matrix:\n", cm)

auc = roc_auc_score(y_test, y_pred)
print("ROC AUC:", auc)

def simulate_degradation(model, sequence):
    preds = model.predict(sequence)
    hli_scores = preds[1]
    return hli_scores
    
from src.data_generator import generate_multimodal_data
from src.preprocessing import preprocess_data
from src.models import build_fusion_model
from src.hli import compute_hli
import numpy as np


def train_pipeline():

    print("Generating Data...")
    df = generate_multimodal_data()

    print("Preprocessing Data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Split modalities
    Xb_train = X_train[:, 0:2]
    Xp_train = X_train[:, 2:4]

    Xb_test = X_test[:, 0:2]
    Xp_test = X_test[:, 2:4]

    print("Building Model...")
    model = build_fusion_model()

    # Create HLI target (normalized degradation severity)
    hli_train = y_train.argmax(axis=1) / 2.0
    hli_test = y_test.argmax(axis=1) / 2.0

    print("Training Model...")
    model.fit(
        [Xb_train, Xp_train],
        {
            "classification": y_train,
            "hli": hli_train
        },
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    print("Evaluating Model...")
    results = model.evaluate(
        [Xb_test, Xp_test],
        {
            "classification": y_test,
            "hli": hli_test
        }
    )

    print("Evaluation Results:", results)

    # Compute HLI on full dataset
    print("Computing Human Limit Index...")
    hli_values, tau = compute_hli(df.drop("label", axis=1).values)

    print("HLI Threshold (tau):", tau)

    return model, hli_values


if __name__ == "__main__":
    train_pipeline()
