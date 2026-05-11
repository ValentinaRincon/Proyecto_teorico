import torch
import numpy as np
import matplotlib.pyplot as plt
from generador_difusion import generador_difusion
from eliminar_ruido import eliminar_ruido

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_timesteps = 200
scheduler = generador_difusion(num_timesteps=num_timesteps)
model = eliminar_ruido(num_timesteps=num_timesteps).to(device)
model.load_state_dict(torch.load("diffusion_1d.pth", map_location=device))
model.eval()

# Parámetros de la distribución objetivo
mu1, mu2 = -2.0, 2.0
sigma1, sigma2 = 0.5, 0.5
weight = 0.5

# Rango de x para graficar
x_min, x_max = -6, 6
x_vals = np.linspace(x_min, x_max, 400)

# Pasos de tiempo a comparar
t_vals = [0, 50, 100, 150, 199]  # t=0 es el paso limpio, pero el score lo calculamos con p_t

# Función para calcular score teórico
def score_teorico(x, t, scheduler):
    alpha_bar = scheduler.alphas_cumprod[t].item()
    var1 = alpha_bar * sigma1**2 + (1 - alpha_bar)
    var2 = alpha_bar * sigma2**2 + (1 - alpha_bar)
    mean1 = np.sqrt(alpha_bar) * mu1
    mean2 = np.sqrt(alpha_bar) * mu2
    
    # pesos no normalizados
    w1 = weight / np.sqrt(2*np.pi*var1) * np.exp(-(x - mean1)**2/(2*var1))
    w2 = weight / np.sqrt(2*np.pi*var2) * np.exp(-(x - mean2)**2/(2*var2))
    total_w = w1 + w2 + 1e-8
    
    score = (w1 * (mean1 - x)/var1 + w2 * (mean2 - x)/var2) / total_w
    return score

# Función para obtener score aprendido por la red
def score_aprendido(x, t, model, scheduler, device):
    # x puede ser array de numpy, lo convertimos a tensor
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.full((len(x),), t, dtype=torch.long).to(device)
    with torch.no_grad():
        noise_pred = model(x_tensor, t_tensor).cpu().numpy().flatten()
    alpha_bar = scheduler.alphas_cumprod[t].item()
    score = - noise_pred / np.sqrt(1 - alpha_bar + 1e-8)
    return score

# Graficar comparación para cada t
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for idx, t in enumerate(t_vals):
    ax = axes[idx]
    score_teo = score_teorico(x_vals, t, scheduler)
    score_apr = score_aprendido(x_vals, t, model, scheduler, device)
    ax.plot(x_vals, score_teo, 'r-', linewidth=2, label='Teórico')
    ax.plot(x_vals, score_apr, 'b--', linewidth=2, label='Aprendido')
    ax.set_title(f't = {t}')
    ax.set_xlabel('x')
    ax.set_ylabel('Fuerza teórica / aprendida')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Si sobra un subplot, lo ocultamos
if len(t_vals) < len(axes):
    axes[-1].axis('off')
fig.suptitle('Comparación de la fuerza teórica vs aprendida', fontsize=16)
plt.tight_layout()
plt.savefig('comparacion_score.png', dpi=150)
plt.show()