import os

import kagglehub
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(AQUI, "modelo_pneumonia_melhorado.keras")
TAMANHO = (150, 150)

caminho = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
base = os.path.join(caminho, "chest_xray")


def carregar(pasta, **kw):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(base, pasta), image_size=TAMANHO, batch_size=32,
        color_mode="grayscale", **kw)


# O conjunto "val" do Kaggle tem só 16 imagens; usamos 20% do train como validação.
treino = carregar("train", validation_split=0.2, subset="training", seed=42)
validacao = carregar("train", validation_split=0.2, subset="validation", seed=42)
teste = carregar("test", shuffle=False)
print("Classes:", treino.class_names)  # ['NORMAL', 'PNEUMONIA']

qtd = np.zeros(2)
for _, y in treino:
    qtd += np.bincount(y.numpy(), minlength=2)
pesos = {i: qtd.sum() / (2 * qtd[i]) for i in range(2)}
print("Pesos das classes:", pesos)


def bloco(filtros):
    return [
        tf.keras.layers.Conv2D(filtros, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
    ]


modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 1)),
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomBrightness(0.2, value_range=(0, 1)),
    tf.keras.layers.RandomContrast(0.2),
    *bloco(32), *bloco(64), *bloco(128), *bloco(128),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(SAIDA, monitor="val_auc", mode="max", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]

modelo.fit(treino, validation_data=validacao, epochs=30,
           class_weight=pesos, callbacks=callbacks)

print("\nAvaliação no conjunto de TESTE (nunca visto no treino):")
modelo.evaluate(teste)
prob = modelo.predict(teste).flatten()
real = np.concatenate([y.numpy() for _, y in teste])
for limiar in (0.5,):
    pred = (prob >= limiar).astype(int)
    print(classification_report(real, pred, target_names=teste.class_names, zero_division=0))
    print(confusion_matrix(real, pred))
print("Modelo salvo em:", SAIDA)
