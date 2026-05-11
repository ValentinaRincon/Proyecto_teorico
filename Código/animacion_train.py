import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from generador_difusion import generador_difusion
from eliminar_ruido import eliminar_ruido
from data_1d import obtener_distribución_objetivo_1d
import numpy as np
import os
from tqdm import tqdm

# Hiperparámetros
batch_size = 1024
num_timesteps = 200
epochs = 50
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_every = 3  # guardar muestras cada 3 épocas

# Crear scheduler y modelo
generador = generador_difusion(num_timesteps=num_timesteps)
model = eliminar_ruido(num_timesteps=num_timesteps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Listas para almacenar información para la animación
epochs_recorded = []
samples_recorded = []  # muestras generadas en esa época
losses_recorded = []   # pérdida media de la época

# Función para generar muestras (sin gradientes)
@torch.no_grad()
def generate_samples(n_samples=2000):
    x = torch.randn(n_samples, 1).to(device)
    for t in reversed(range(num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_tensor)
        beta_t = generador.betas[t]
        alpha_t = generador.alphas[t]
        alpha_cumprod_t = generador.alphas_cumprod[t]
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        mean = coef1 * (x - coef2 * noise_pred)
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
    return x.cpu().numpy()

print("Comenzando entrenamiento...")
for epoch in range(epochs):
    total_loss = 0
    # Usamos tqdm para barra de progreso en cada época
    pbar = tqdm(range(100), desc=f"Época {epoch+1}/{epochs}")
    for step in pbar:
        x_0 = obtener_distribución_objetivo_1d(batch_size, "mixture").to(device)
        t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_0)
        x_t = generador.q_sample(x_0, t, noise)
        noise_pred = model(x_t, t)
        loss = loss_fn(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / 100
    print(f"Época {epoch+1} completada, pérdida media: {avg_loss:.6f}")
    
    # Guardar información si corresponde
    if (epoch + 1) % save_every == 0 or epoch == 0 or epoch == epochs-1:
        print(f"  → Generando muestras para época {epoch+1}...")
        samples = generate_samples(n_samples=2000)
        epochs_recorded.append(epoch + 1)
        samples_recorded.append(samples)
        losses_recorded.append(avg_loss)

# Guardar modelo final
torch.save(model.state_dict(), "diffusion_1d.pth")
print("Entrenamiento finalizado. Creando animación...")

# --- Crear animación ---
fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
ax_hist = axes[0]
ax_loss = axes[1]

# Configurar histograma
ax_hist.set_xlim(-6, 6)
ax_hist.set_ylim(0, 0.8)
ax_hist.set_xlabel("Valor")
ax_hist.set_ylabel("Densidad")
ax_hist.set_title("Evolución de la distribución generada durante el entrenamiento")
n_bins = 80
# Histograma generado (azul)
hist = ax_hist.hist([], bins=n_bins, range=(-6,6), density=True, alpha=0.7, color='blue', label='Generado')[2]
# Distribución real (rojo fijo)
x_real = obtener_distribución_objetivo_1d(5000, "mixture").cpu().numpy()
ax_hist.hist(x_real, bins=n_bins, range=(-6,6), density=True, alpha=0.3, color='red', label='Real')
ax_hist.legend(loc='upper right')

# Texto para mostrar época y loss
info_text = ax_hist.text(0.02, 0.95, '', transform=ax_hist.transAxes, fontsize=12, color='blue')

# Configurar gráfico de pérdida
ax_loss.set_xlim(0, len(epochs_recorded)-1)
ax_loss.set_ylim(0, max(losses_recorded)*1.1)
ax_loss.set_xlabel("Época guardada (índice)")
ax_loss.set_ylabel("Pérdida (MSE)")
loss_line, = ax_loss.plot([], [], 'o-', color='green', linewidth=2, markersize=4)
ax_loss.grid(True, alpha=0.3)

def init():
    for patch in hist:
        patch.set_height(0)
    info_text.set_text('')
    loss_line.set_data([], [])
    return list(hist) + [info_text, loss_line]

def update(frame):
    # frame es el índice en la lista de épocas guardadas
    epoch_num = epochs_recorded[frame]
    samples = samples_recorded[frame]
    loss_val = losses_recorded[frame]
    
    # Actualizar histograma
    counts, bins = np.histogram(samples.flatten(), bins=n_bins, range=(-6,6), density=True)
    for count, patch in zip(counts, hist):
        patch.set_height(count)
    
    # Actualizar texto
    info_text.set_text(f'Época {epoch_num}   Pérdida: {loss_val:.4f}')
    
    # Actualizar gráfico de pérdida (mostrar toda la serie hasta el frame actual)
    x_data = list(range(frame+1))
    y_data = losses_recorded[:frame+1]
    loss_line.set_data(x_data, y_data)
    ax_loss.relim()
    ax_loss.autoscale_view()
    
    return list(hist) + [info_text, loss_line]

ani = animation.FuncAnimation(fig, update, frames=len(epochs_recorded),
                              init_func=init, blit=True, interval=500)  # 500 ms entre frames

# Guardar animación
ani.save('entrenamiento_difusion.gif', writer='pillow', fps=5)  # fps bajo para que se vea pausado
print("Animación guardada como 'entrenamiento_difusion.gif'")