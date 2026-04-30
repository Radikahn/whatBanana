from tensorflow.keras.models import load_model
from evaluation.metrics import SparsePrecision, SparseRecall, SparseF1Score

model = load_model('best_cnn_model.keras', custom_objects={
    'SparseRecall': SparseRecall,
    'SparsePrecision': SparsePrecision,
    'SparseF1Score': SparseF1Score,
})

model.summary()