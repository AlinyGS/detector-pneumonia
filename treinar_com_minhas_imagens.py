"""Treina o detector de pneumonia com as imagens do usuário (transfer learning MobileNetV2).

- Divide treino/validação/teste POR PACIENTE (evita a mesma pessoa em dois conjuntos,
  o que inflaria a acurácia).
- Escolhe o limiar de decisão na validação e salva em limiar.json.
"""
import json
import os
import re

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = r"C:\Users\julio\Downloads\Modelo de Imagem - Máquinas Ensináveis_files"
SAIDA = os.path.join(AQUI, "modelo_pneumonia_v2.keras")
TAM = 224
tf.keras.utils.set_random_seed(42)

# ---------- dados ----------
arquivos, rotulos, grupos = [], [], []
for rotulo, classe in enumerate(["NORMAL", "PNEUMONIA"]):
    pasta = os.path.join(RAIZ, classe)
    for nome in sorted(os.listdir(pasta)):
        if not nome.lower().endswith((".jpeg", ".jpg", ".png")):
            continue
        m = re.match(r"(person\d+)", nome) or re.match(r"(.*?IM-\d+)", nome)
        arquivos.append(os.path.join(pasta, nome))
        rotulos.append(rotulo)
        grupos.append(m.group(1) if m else nome)
arquivos, rotulos, grupos = map(np.array, (arquivos, rotulos, grupos))
print("Imagens:", len(arquivos), "| normais:", (rotulos == 0).sum(), "| pneumonia:", (rotulos == 1).sum())

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
ini, te = next(gss.split(arquivos, rotulos, grupos))
gss = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
tr, va = next(gss.split(arquivos[ini], rotulos[ini], grupos[ini]))
tr, va = ini[tr], ini[va]
print("treino/val/teste:", len(tr), len(va), len(te))


def carregar(caminho, y):
    img = tf.io.decode_jpeg(tf.io.read_file(caminho), channels=3)
    img = tf.image.resize(img, (TAM, TAM))
    return img, tf.cast(y, tf.float32)


def dataset(idx, treino=False):
    ds = tf.data.Dataset.from_tensor_slices((arquivos[idx], rotulos[idx]))
    if treino:
        ds = ds.shuffle(len(idx), seed=42)
    ds = ds.map(carregar, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    return ds.prefetch(tf.data.AUTOTUNE)


ds_tr, ds_va, ds_te = dataset(tr, True), dataset(va), dataset(te)

n0, n1 = (rotulos[tr] == 0).sum(), (rotulos[tr] == 1).sum()
pesos = {0: len(tr) / (2 * n0), 1: len(tr) / (2 * n1)}
print("Pesos:", pesos)

# ---------- modelo ----------
aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
])
base = tf.keras.applications.MobileNetV2(input_shape=(TAM, TAM, 3), include_top=False, weights="imagenet")
base.trainable = False

entrada = tf.keras.Input(shape=(TAM, TAM, 3))
x = aug(entrada)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # [-1, 1]
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
saida = tf.keras.layers.Dense(1, activation="sigmoid")(x)
modelo = tf.keras.Model(entrada, saida)

metricas = ["accuracy", tf.keras.metrics.AUC(name="auc")]


def cb():
    return [
        tf.keras.callbacks.ModelCheckpoint(SAIDA, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]


print("\n== Fase 1: treinando só a cabeça ==")
modelo.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=metricas)
modelo.fit(ds_tr, validation_data=ds_va, epochs=8, class_weight=pesos, callbacks=cb())

print("\n== Fase 2: ajuste fino das últimas camadas ==")
base.trainable = True
for camada in base.layers[:-40]:
    camada.trainable = False
modelo.compile(tf.keras.optimizers.Adam(1e-5), "binary_crossentropy", metrics=metricas)
modelo.fit(ds_tr, validation_data=ds_va, epochs=15, class_weight=pesos, callbacks=cb())

# ---------- limiar (escolhido na validação) ----------
p_va = modelo.predict(ds_va).flatten()
y_va = rotulos[va]
melhor, melhor_j = 0.5, -1
for t in np.linspace(0.05, 0.95, 91):
    pred = p_va >= t
    sens = (pred & (y_va == 1)).sum() / (y_va == 1).sum()
    espec = (~pred & (y_va == 0)).sum() / (y_va == 0).sum()
    if sens + espec - 1 > melhor_j:  # índice de Youden
        melhor, melhor_j = float(t), sens + espec - 1
print("Limiar escolhido:", melhor)
with open(os.path.join(AQUI, "limiar.json"), "w") as f:
    json.dump({"limiar": melhor, "tamanho": TAM}, f)

# ---------- teste final ----------
p_te = modelo.predict(ds_te).flatten()
y_te = rotulos[te]
pred = (p_te >= melhor).astype(int)
print("\n=== TESTE (pacientes nunca vistos) ===")
print("AUC:", roc_auc_score(y_te, p_te))
print(classification_report(y_te, pred, target_names=["NORMAL", "PNEUMONIA"], digits=3))
print(confusion_matrix(y_te, pred))
print("Modelo salvo em:", SAIDA)
