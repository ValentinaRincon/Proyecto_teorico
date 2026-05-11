import torch
import numpy as np
import matplotlib.pyplot as plt
from generador_difusion import generador_difusion
from data_1d import obtener_distribución_objetivo_1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_timesteps = 200
scheduler = generador_difusion(num_timesteps=num_timesteps)
batch_size = 5000
x_0 = obtener_distribución_objetivo_1d(batch_size, "mixture").to(device)

t_steps = [0, 50, 100, 150, 199]
colors = ['#D1E0FF', '#E4FFD1', '#FFE2D1', '#FFD6D1', 'purple']
bins = np.linspace(-6, 6, 60)

plt.figure(figsize=(10, 6))
for t, color in zip(t_steps, colors):
    t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
    x_t = scheduler.q_sample(x_0, t_tensor).cpu().numpy().flatten()
    plt.hist(x_t, bins=bins, density=True, alpha=0.5, color=color,
             edgecolor='black', linewidth=0.5, label=f't = {t}')

plt.title('Evolución Forward')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.xlim(-6, 6)
plt.ylim(0, 0.6)
plt.legend()
plt.grid(alpha=0.2)
#plt.savefig('forward_superpuesto_con_borde.png', dpi=150)
plt.show()