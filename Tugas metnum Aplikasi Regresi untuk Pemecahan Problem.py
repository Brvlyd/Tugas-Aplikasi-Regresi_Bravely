import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Impor data dari file CSV
file_path = '/mnt/data/Student_Performance.csv'
data = pd.read_csv(file_path)

# Extract relevant columns
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values

# Reshape NL for sklearn
NL_reshaped = NL.reshape(-1, 1)

# Model Linear
linear_model = LinearRegression()
linear_model.fit(NL_reshaped, NT)
NT_pred_linear = linear_model.predict(NL_reshaped)

# Model Eksponensial
# y = Ce^(bX) => log(y) = log(C) + b*X
log_NT = np.log(NT)
exp_model = LinearRegression()
exp_model.fit(NL_reshaped, log_NT)
log_C = exp_model.intercept_
b = exp_model.coef_[0]
C = np.exp(log_C)
NT_pred_exp = C * np.exp(b * NL)

# Plot data dan hasil regresi
plt.figure(figsize=(14, 6))

# Plot Model Linear
plt.subplot(1, 2, 1)
plt.scatter(NL, NT, color='blue', label='Data Asli')
plt.plot(NL, NT_pred_linear, color='red', label='Regresi Linear')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear')
plt.legend()

# Plot Model Eksponensial
plt.subplot(1, 2, 2)
plt.scatter(NL, NT, color='blue', label='Data Asli')
plt.plot(NL, NT_pred_exp, color='green', label='Regresi Eksponensial')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Eksponensial')
plt.legend()

plt.tight_layout()
plt.show()

# Hitung galat RMS
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
rms_exp = np.sqrt(mean_squared_error(NT, NT_pred_exp))

print(f"RMS galat - Regresi Linear: {rms_linear}")
print(f"RMS galat - Regresi Eksponensial: {rms_exp}")
