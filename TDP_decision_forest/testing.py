import pandas as pd

test_df = pd.read_csv("https://github.com/joao-arroyo/Datos/blob/144512fcd29a05f0c789cf596250182f3c2cf165/Research_data_SAMS-test2-ok-test.csv?raw=true")

X = test_df.iloc[:, 0:26]
Y = test_df.iloc[:, 28]
y_test = Y

