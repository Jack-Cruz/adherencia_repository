import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

train_df = pd.read_csv("https://github.com/joao-arroyo/Datos/blob/144512fcd29a05f0c789cf596250182f3c2cf165/Research_data_SAMS-test2-ok-training.csv?raw=true")
test_df = pd.read_csv("https://github.com/joao-arroyo/Datos/blob/144512fcd29a05f0c789cf596250182f3c2cf165/Research_data_SAMS-test2-ok-test.csv?raw=true")

X = train_df.iloc[:, 0:26]
Y = train_df.iloc[:, 28]
train_df = X
train_df["result"] = Y

X = test_df.iloc[:, 0:26]
Y = test_df.iloc[:, 28]
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