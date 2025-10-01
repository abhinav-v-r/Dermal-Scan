# Build Model (MobileNetV2)
# ===============================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze convolutional base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation="softmax")
])

# ===============================
# Compile
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# Train
# ===============================
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)