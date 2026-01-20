from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# =====================
# KONFIGURASI
# =====================
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 26
LR_INITIAL = 3e-4

# =====================
# BASE MODEL
# =====================
base_model = MobileNetV2(
    input_shape=INPUT_SHAPE,
    include_top=False,
    weights="imagenet"
)

for layer in base_model.layers:
    layer.trainable = False

# =====================
# CLASSIFIER
# =====================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LR_INITIAL),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
