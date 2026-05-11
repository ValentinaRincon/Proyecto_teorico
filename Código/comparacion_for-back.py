import torch
import numpy as np
import matplotlib.pyplot as plt
from generador_difusion import generador_difusion
from eliminar_ruido import eliminar_ruido
from data_1d import obtener_distribución_objetivo_1d

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_timesteps = 200
scheduler = generador_difusion(num_timesteps=num_timesteps)

# Cargar modelo entrenado
model = eliminar_ruido(num_timesteps=num_timesteps).to(device)
model.load_state_dict(torch.load("diffusion_1d.pth", map_location=device))
model.eval()

# 1. Generar muestras con el modelo (reverse completo)
def sample(model, scheduler, n_samples=5000):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, 1).to(device)
        for t in reversed(range(num_timesteps)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(x, t_tensor)
            beta_t = scheduler.betas[t]
            alpha_t = scheduler.alphas[t]
            alpha_cumprod_t = scheduler.alphas_cumprod[t]
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            mean = coef1 * (x - coef2 * noise_pred)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return x.cpu().numpy().flatten()

n_samples = 5000
generated_samples = sample(model, scheduler, n_samples)

# 2. Obtener muestras reales de la distribución objetivo
real_samples = obtener_distribución_objetivo_1d(n_samples, "mixture").cpu().numpy().flatten()

# 3. Graficar histogramas superpuestos
bins = np.linspace(-6, 6, 80)
plt.figure(figsize=(8, 5))
plt.hist(real_samples, bins=bins, density=True, alpha=0.6, color='red',
         edgecolor='black', linewidth=0.5, label='Distribución real')
plt.hist(generated_samples, bins=bins, density=True, alpha=0.6, color='blue',
         edgecolor='black', linewidth=0.5, label='Distribución generada')

plt.title('Distribución real vs generada')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.xlim(-6, 6)
plt.ylim(0, 0.6)
plt.legend()
plt.grid(alpha=0.3)
#plt.savefig('comparacion_real_vs_generada.png', dpi=150)
plt.show()