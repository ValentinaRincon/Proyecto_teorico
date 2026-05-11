import torch
import numpy as np
import matplotlib.pyplot as plt
from generador_difusion import generador_difusion
from eliminar_ruido import eliminar_ruido

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_timesteps = 200
scheduler = generador_difusion(num_timesteps=num_timesteps)

model = eliminar_ruido(num_timesteps=num_timesteps).to(device)
model.load_state_dict(torch.load("diffusion_1d.pth", map_location=device))
model.eval()

n_samples = 5000
t_steps = [199, 150, 100, 50, 0]   # pasos que queremos guardar

def get_intermediate_samples(model, scheduler, n_samples, t_steps):
    samples = {}
    with torch.no_grad():
        x = torch.randn(n_samples, 1).to(device)
        if 199 in t_steps:
            samples[199] = x.cpu().numpy().flatten()
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
            if (t-1) in t_steps:
                samples[t-1] = x.cpu().numpy().flatten()
        return samples

samples_dict = get_intermediate_samples(model, scheduler, n_samples, t_steps)

bins = np.linspace(-6, 6, 60)
colors = ['#FBD1FF', '#FFD6D1', '#FFE2D1', '#E4FFD1', 'blue']  # orden: 199,150,100,50,0

plt.figure(figsize=(10, 6))
for t, color in zip(t_steps, colors):
    data = samples_dict[t]
    plt.hist(data, bins=bins, density=True, alpha=0.5, color=color,
             edgecolor='black', linewidth=0.5, label=f't = {t}')

plt.title('Evolución Backward')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.xlim(-6, 6)
plt.ylim(0, 0.6)
plt.legend()
plt.grid(alpha=0.2)
#plt.savefig('reverse_superpuesto_barras.png', dpi=150)
plt.show()