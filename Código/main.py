import torch
import matplotlib.pyplot as plt
from generador_difusion import generador_difusion
from data_1d import obtener_distribución_objetivo_1d

# 1. Configuración inicial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generador = generador_difusion(num_timesteps=200)  # menos pasos para ver cambios rápido

# 2. Generamos datos limpios
batch_size = 2000
x_0 = obtener_distribución_objetivo_1d(batch_size, "mixture").to(device)

# 3. Elegimos algunos pasos de tiempo para ver el proceso
t_steps = [0, 50, 100, 150, 199]  # t=0: datos limpios; t=199: casi ruido

# 4. Aplicamos forward para cada t
x_t_dict = {}
for t_val in t_steps:
    # Creamos un tensor de t's, todos iguales a t_val
    t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
    # Aplicamos la fórmula
    x_t = generador.q_sample(x_0, t)
    x_t_dict[t_val] = x_t.cpu().numpy()  # pasamos a numpy para graficar

# 5. Dibujamos histogramas
fig, axes = plt.subplots(1, len(t_steps), figsize=(15, 3))
for i, (t_val, x_t) in enumerate(x_t_dict.items()):
    axes[i].hist(x_t, bins=50, density=True, alpha=0.7, color='blue')
    axes[i].set_title(f"Paso t={t_val}")
    axes[i].set_xlim(-8, 8)  # fijamos límites para comparar
plt.tight_layout()
plt.show()