from tensorflow.keras.models import load_model
from model.evaluation.metrics import SparsePrecision, SparseRecall, SparseF1Score
from loader.banana import init_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def test_model(model_name, model_path):
    _,_, test_ds = init_data(img_size=(224, 224), batch_size=32)

    # Load best model
    print(f"Loading best {model_name} model...")
    model = load_model(model_path, custom_objects={
        'SparseRecall': SparseRecall,
        'SparsePrecision': SparsePrecision,
        'SparseF1Score': SparseF1Score,
    })
    print("Model loaded successfully!")

    # Summary
    print("\nModel Summary:")
    model.summary()

    # Evaluate
    print("\nEvaluating on test dataset...")
    results = model.evaluate(test_ds)

    with open(f"{model_name}_metrics.txt", "w") as f:
            for name, value in zip(model.metrics_names, results):
                line = f"{name}: {value}\n"
                print(line.strip())
                f.write(line)

    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_ds)
    pred_labels = np.argmax(predictions, axis=1)

    # Get true labels
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=test_ds.class_names))

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_ds.class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()