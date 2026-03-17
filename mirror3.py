import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import keras
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import json

# --- Lista dei dataset --- #
excel_files = ["Mg.xlsx", "Mg_f1.xlsx", "OH.xlsx", "OH_f1.xlsx", "DFSB.xlsx", "DFSB_c1.xlsx"] # datasets 

# Hyperparameters
Dense_1 = 176
Dense_2 = 240
Dense_3 = 64
Dense_4 = 64
Dense_5 = 64
Dense_6 = 64

learning_rate = 1e-3
batch_size = 60
epochs = 2000

# Array per accumulare tutti i dataset
input_all, output_all = [], []

# Array per accumulare tutti i dataset
x_train_all, y_train_all = [], []
x_test_all, y_test_all = [], []

# Scaling globale
scaler_X_global = MinMaxScaler(feature_range=(0,1))
scaler_y_global = MinMaxScaler(feature_range=(0,1))

# --- Preprocessing e split dei dataset --- #
for file in excel_files:
    df = pd.read_excel(file)

    # --- FILTRO SUI DIAMETRI --- #
    condition = (df["d10"] < df["d21"]) & (df["d21"] < df["d32"]) & (df["d32"] < df["d43"])
    df_filtered = df[condition].copy()
    
    print(f"{file}: rimosse {len(df) - len(df_filtered)} simulazioni non valide.")
    
    df = df_filtered
    
    # Separazione input/output
    input_df = df.drop(['SimulationID', 'A1', 'B1', 'kg', 'g', 'C_adjust', 'Ap'], axis=1)
    output_df = df.drop(['SimulationID', 'MgCl2Inlet', 'NaOHInlet', 'VdotMg', 'VdotOH', 'Mg0', 'OH0', 'feedingTime', 'd10', 'd21', 'd32', 'd43'], axis=1)
    
    input_array = input_df.to_numpy()
    output_array = output_df.to_numpy()
    
    # Log10 su colonne specifiche
    cols_input_log = [7,8,9,10]
    cols_output_log = [0,2] # A, kg, Ap
    
    # Controllo valori <= 0
    mask_invalid = input_array[:, cols_input_log] <= 0
    if np.any(mask_invalid):
        bad_indices = np.argwhere(mask_invalid)
        print(f"⚠️ Valori <= 0 in {file} alle posizioni (riga, colonna):")
        for row, col in bad_indices:
            print(f"  → Riga {row}, colonna {cols_input_log[col]} (valore: {input_array[row, cols_input_log[col]]})")
    
    input_array[:, cols_input_log] = np.log10(input_array[:, cols_input_log])
    output_array[:, cols_output_log] = np.log10(output_array[:, cols_output_log])
    
    input_all.append(input_array)
    output_all.append(output_array)

# Converti la lista di array in un singolo array 2D
input_all = np.vstack(input_all)
output_all = np.vstack(output_all)

# Scaling
input_scaled = scaler_X_global.fit_transform(input_all)
output_scaled = scaler_y_global.fit_transform(output_all)

# Split training/testing (0.8/0.2)
x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(input_scaled, output_scaled, test_size=0.1, random_state=42)

# Concatenazione dei dataset
x_train_all = np.vstack(x_train_all)
y_train_all = np.vstack(y_train_all)
x_test_all = np.vstack(x_test_all)
y_test_all = np.vstack(y_test_all)

print("Shape training:", x_train_all.shape, y_train_all.shape)
print("Shape testing:", x_test_all.shape, y_test_all.shape)

# %%

# --- Definizione del modello --- #
model = Sequential()
initializer = 'normal'
model.add(Dense(Dense_1, input_dim=x_train_all.shape[1], kernel_initializer=initializer, activation='relu'))
model.add(Dense(Dense_2, kernel_initializer=initializer, activation='relu'))
model.add(Dense(Dense_3, kernel_initializer=initializer, activation='relu'))
model.add(Dense(Dense_4, kernel_initializer=initializer, activation='relu'))
model.add(Dense(Dense_5, kernel_initializer=initializer, activation='relu'))
model.add(Dense(Dense_6, kernel_initializer=initializer, activation='relu'))
model.add(Dense(y_train_all.shape[1], kernel_initializer=initializer, activation='linear'))

model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

# --- Callbacks --- #
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_loss', verbose=1, patience=5000)

class PlotCurrentEstimate(Callback):
    def __init__(self, update_freq=2):
        super().__init__()
        self.epoch = 0
        self.update_freq = update_freq
        self.h = {'loss': [], 'val_loss': []}
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        if self.epoch % self.update_freq == 0:
            clear_output(wait=True)
            self.h['loss'].append(logs['loss'])
            self.h['val_loss'].append(logs['val_loss'])

trainplot = PlotCurrentEstimate()

# --- Training --- #
history = model.fit(x_train_all, y_train_all,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test_all, y_test_all),
                    callbacks=[checkpoint, es, trainplot])

# %%

# --- Plot training history --- #
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.yscale("log")
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("trainingCurve.png")
plt.show()

# --- Valutazione modello --- #
y_pred_train = scaler_y_global.inverse_transform(model.predict(x_train_all))
y_train_rescale = scaler_y_global.inverse_transform(y_train_all)

y_pred_test = scaler_y_global.inverse_transform(model.predict(x_test_all))
y_test_rescale = scaler_y_global.inverse_transform(y_test_all)

errore = np.abs((y_test_rescale - y_pred_test) / y_test_rescale) * 100
errore_medio = np.mean(errore, axis=0)

print(f'Average percentage error on test set = {errore_medio}')
print(f'Maximum percentage error = {np.max(errore)}')
print(f'Minimum percentage error = {np.min(errore)}')

# %%

# --- Plot predizioni --- #
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs = axs.ravel()
for i in range(y_test_rescale.shape[1]):
    axs[i].plot(y_test_rescale[:, i], y_pred_test[:, i], 'ro')
    axs[i].plot(y_test_rescale[:, i], y_test_rescale[:, i], 'k--')
fig.savefig("testing.png")
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs = axs.ravel()
for i in range(y_train_rescale.shape[1]):
    axs[i].plot(y_train_rescale[:, i], y_pred_train[:, i], 'bo')
    axs[i].plot(y_train_rescale[:, i], y_train_rescale[:, i], 'k--')
fig.savefig("training.png")
plt.show()

# %%


# --- Carica il miglior modello salvato --- #
model = load_model('best_model.keras')

# --- Predizioni su nuovi diametri --- #

# [0.25, 0.0, 0.5*(0.001/60), 0.0,
# 0.0, 0.072, 12000, 89.0, 140.8, 323.0, 567.1], # OH_f1

# [0.5, 1, 0.5*(0.001/60), 0.5*(0.001/60),
# 0.0, 0.0, 6000, 213, 244, 286, 339],           # DFSB

# [0.125, 0.25, 0.5*(0.001/60), 0.5*(0.001/60),
# 0.0, 0.0, 6000, 227, 259, 302, 356]            # DFSB_c1

diameters = np.array([
    [0, 0.5, 0.0, 0.0000166666666666667,
    0.036, 0.0, 6000, 73.7, 82.9, 106.2, 204.2],

    [0, 0.5, 0.0, 0.5*(0.001/60),
    0.036, 0.0, 12000, 102.3, 118.6, 144.5, 181.6],

    [0.25, 0.0, 0.0000166666666666667, 0.0,
    0.0, 0.072, 6000, 58.9, 68.5, 89.5, 138.8],
])

# Applica log10 solo alle ultime 4 posizioni
diameters[:, -4:] = np.log10(diameters[:, -4:])

diameters_scaled = scaler_X_global.transform(diameters)
kinetics_predictions = model.predict(diameters_scaled)
kinetics_predictions = scaler_y_global.inverse_transform(kinetics_predictions)

mean_values = np.mean(kinetics_predictions, axis=0)
std_values = np.std(kinetics_predictions, axis=0)

print("Mean values per column:", mean_values)
print("Standard deviation per column:", std_values)

# Salvataggio statistiche in JSON
statistics = {"mean_values": mean_values.tolist(), "std_values": std_values.tolist()}
with open("kinetics_statistics.json", "w") as f:
    json.dump(statistics, f, indent=4)
print("File 'kinetics_statistics.json' salvato correttamente!")

# %%

# import keras_tuner as kt

# def build_model(hp):
#     model = Sequential()
#     initializer = 'normal'
    
#     # Scelta del numero di neuroni per ogni layer
#     model.add(Dense(hp.Int('units_1', min_value=16, max_value=256, step=16), 
#                     input_dim=x_train_all.shape[1], 
#                     kernel_initializer=initializer, 
#                     activation='relu'))
#     model.add(Dense(hp.Int('units_2', min_value=16, max_value=256, step=16), 
#                     kernel_initializer=initializer, 
#                     activation='relu'))
#     model.add(Dense(hp.Int('units_3', min_value=16, max_value=256, step=16), 
#                     kernel_initializer=initializer, 
#                     activation='relu'))
#     model.add(Dense(hp.Int('units_4', min_value=16, max_value=256, step=16), 
#                     kernel_initializer=initializer, 
#                     activation='relu'))
#     model.add(Dense(hp.Int('units_5', min_value=16, max_value=256, step=16), 
#                     kernel_initializer=initializer, 
#                     activation='relu'))
#     model.add(Dense(hp.Int('units_6', min_value=16, max_value=256, step=16), 
#                     kernel_initializer=initializer, 
#                     activation='relu'))
#     model.add(Dense(y_train_all.shape[1], kernel_initializer=initializer, activation="linear"))
    
#     # Scelta del tasso di apprendimento
#     learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 1e-5])
#     model.compile(loss=keras.losses.mean_absolute_error,
#                   optimizer=Adam(learning_rate=learning_rate),
#                   metrics=['accuracy'])
    
#     return model

# # Ricerca degli iperparametri con Hyperband
# tuner = kt.Hyperband(build_model,
#                      objective='val_loss',
#                      max_epochs=100,
#                      factor=3,
#                      directory='my_tuner_results',
#                      project_name='hyperparam_opt')

# # Esecuzione della ricerca
# tuner.search(x_train_all, y_train_all,
#              epochs=10000,
#              validation_data=(x_test_all, y_test_all),
#              callbacks=[EarlyStopping(monitor='val_loss', patience=1000)])

# # Miglior modello trovato
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"Migliori iperparametri trovati: ")
# print(f"Units layer 1: {best_hps.get('units_1')}")
# print(f"Units layer 2: {best_hps.get('units_2')}")
# print(f"Units layer 3: {best_hps.get('units_3')}")
# print(f"Units layer 4: {best_hps.get('units_3')}")
# print(f"Units layer 5: {best_hps.get('units_3')}")
# print(f"Units layer 6: {best_hps.get('units_3')}")
# print(f"Learning rate: {best_hps.get('learning_rate')}")
