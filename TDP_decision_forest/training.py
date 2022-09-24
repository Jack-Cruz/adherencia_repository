import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

train_df = pd.read_csv("https://github.com/joao-arroyo/Datos/blob/434950ffbee0190ebb5ea8ab8b68bedf9448299e/DATA_TRAIN2.csv?raw=true")
test_df = pd.read_csv("https://github.com/joao-arroyo/Datos/blob/434950ffbee0190ebb5ea8ab8b68bedf9448299e/DATA_TEST2.csv?raw=true")

X = train_df.iloc[:, 0:14]
Y = train_df.iloc[:, 17]
train_df = X
train_df["result"] = Y

X = test_df.iloc[:, 0:14]
Y = test_df.iloc[:, 17]
y_test = Y
test_df = X
test_df["result"] = Y

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="result")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="result")

# Train a Random Forest model.
model = tfdf.keras.RandomForestModel()
model.fit(x=train_ds)

# Summary of the model structure.
model.summary()
model.compile(metrics=["accuracy"])

# Evaluate model
evaluation = model.evaluate(test_ds)
print(f"Binary Cross entropy loss: {evaluation[0]}")
print(f"Accuracy: {evaluation[1]}")

# Export the model
print('Saving...')
model.save("./tmp/adherencia")

model.make_inspector().variable_importances()