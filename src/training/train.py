from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)
]

base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks
)