import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from generador_difusion import generador_difusion
from eliminar_ruido import eliminar_ruido
from data_1d import obtener_distribución_objetivo_1d
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_timesteps = 200
scheduler = generador_difusion(num_timesteps=num_timesteps)

model = eliminar_ruido(num_timesteps=num_timesteps).to(device)
model.load_state_dict(torch.load("diffusion_1d.pth", map_location=device))
model.eval()

n_samples = 2000
# Generar ruido inicial
x = torch.randn(n_samples, 1).to(device)

# Preparar frames: queremos mostrar desde t=199 hasta t=0, pero guardando algunos intermedios
steps_to_show = list(range(0, num_timesteps, 4))  # cada 4 pasos
if 0 not in steps_to_show:
    steps_to_show.append(0)
if num_timesteps-1 not in steps_to_show:
    steps_to_show.append(num_timesteps-1)
steps_to_show = sorted(steps_to_show, reverse=True)  # orden inverso: de T a 0

# Función que genera muestras en los pasos deseados
def generate_samples_at_steps(steps):
    samples = {}
    with torch.no_grad():  # Desactiva gradientes automáticos
        x_t = torch.randn(n_samples, 1).to(device)
        current_t = num_timesteps - 1
        samples[current_t] = x_t.cpu().numpy()  # ahora funciona sin error
        
        for t in reversed(range(num_timesteps)):
            if t == 0:
                continue
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(x_t, t_tensor)
            beta_t = scheduler.betas[t]
            alpha_t = scheduler.alphas[t]
            alpha_cumprod_t = scheduler.alphas_cumprod[t]
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            mean = coef1 * (x_t - coef2 * noise_pred)
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(beta_t) * noise
            
            if t-1 in steps:
                samples[t-1] = x_t.cpu().numpy()  # ya seguro
        
        # Aseguramos t=0
        if 0 in steps and 0 not in samples:
            samples[0] = x_t.cpu().numpy()
    return samples

print("Generando muestras para reverse...")
samples_dict = generate_samples_at_steps(steps_to_show)

# Figura
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-6, 6)
ax.set_ylim(0, 0.8)
ax.set_xlabel("Valor")
ax.set_ylabel("Densidad")
ax.set_title("Proceso backward")
n_bins = 80
hist = ax.hist([], bins=n_bins, range=(-6,6), density=True, alpha=0.7, color='blue')[2]
# Distribución real (fija)
x_real = obtener_distribución_objetivo_1d(5000, "mixture").cpu().numpy()
ax.hist(x_real, bins=n_bins, range=(-6,6), density=True, alpha=0.3, color='red', label='Real')
ax.legend()

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='blue')

def init():
    for patch in hist: patch.set_height(0)
    time_text.set_text('')
    return list(hist) + [time_text]

def update(frame):
    t = steps_to_show[frame]  # t va de T a 0
    data = samples_dict[t].flatten()
    counts, bins = np.histogram(data, bins=n_bins, range=(-6,6), density=True)
    for count, patch in zip(counts, hist): patch.set_height(count)
    time_text.set_text(f'Paso t = {t}  (progreso: {100*(1 - t/num_timesteps):.1f}%)')
    return list(hist) + [time_text]

ani = animation.FuncAnimation(fig, update, frames=len(steps_to_show),
                              init_func=init, blit=True, interval=100)
ani.save('proceso_backward.gif', writer='pillow', fps=10)
print("Guardado.")