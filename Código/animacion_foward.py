import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from generador_difusion import generador_difusion
from data_1d import obtener_distribución_objetivo_1d
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scheduler = generador_difusion(num_timesteps=200)

# Datos limpios
n_samples = 2000
x_0 = obtener_distribución_objetivo_1d(n_samples, "mixture").to(device)

# Pasos a mostrar
steps_to_show = list(range(0, scheduler.num_timesteps, 4))  # cada 4 pasos
if 0 not in steps_to_show:
    steps_to_show.insert(0, 0)
if scheduler.num_timesteps-1 not in steps_to_show:
    steps_to_show.append(scheduler.num_timesteps-1)
steps_to_show = sorted(steps_to_show)

# Generar datos para cada paso
samples_dict = {}
for t_val in steps_to_show:
    t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
    x_t = scheduler.q_sample(x_0, t)
    samples_dict[t_val] = x_t.cpu().numpy()

# Animación similar
fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(-6, 6); ax.set_ylim(0, 0.8)
ax.set_xlabel("Valor"); ax.set_ylabel("Densidad")
ax.set_title("Proceso forward (añadiendo ruido)")
n_bins = 80
hist = ax.hist([], bins=n_bins, range=(-6,6), density=True, alpha=0.7, color='blue')[2]
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

def init():
    for patch in hist: patch.set_height(0)
    time_text.set_text('')
    return list(hist) + [time_text]

def update(frame):
    t = steps_to_show[frame]
    data = samples_dict[t].flatten()
    counts, bins = np.histogram(data, bins=n_bins, range=(-6,6), density=True)
    for count, patch in zip(counts, hist): patch.set_height(count)
    time_text.set_text(f'Paso t = {t}')
    return list(hist) + [time_text]

ani = animation.FuncAnimation(fig, update, frames=len(steps_to_show),
                              init_func=init, blit=True, interval=100)
ani.save('animacion_forward.gif', writer='pillow', fps=10)
print("Animación forward guardada.")