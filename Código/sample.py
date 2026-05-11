import torch
import matplotlib.pyplot as plt
from generador_difusion import generador_difusion
from eliminar_ruido import eliminar_ruido
from data_1d import obtener_distribución_objetivo_1d

"""
Empezamos con ruido puro (normal estándar).
Para cada t desde el último hasta el primero, usamos el modelo para predecir el ruido que había en ese paso.
Con esa predicción, calculamos la media de la distribución de x_{t-1} (según las ecuaciones de DDPM).
Luego, si no es el último paso, añadimos un poco de ruido (la varianza es beta_t) para mantener la estocasticidad.
Al final, t=0, obtenemos x_0 que debería parecerse a los datos reales.
"""

def sample(model, scheduler, n_samples=1000, device='cpu'):
    model.eval()  # modo evaluación (desactiva dropout, etc.)
    with torch.no_grad():  # no necesitamos gradientes
        # Empezamos con ruido puro
        x = torch.randn(n_samples, 1).to(device)
        
        # Vamos de t = T-1 hasta 0
        for t in reversed(range(scheduler.num_timesteps)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Predicción del ruido
            noise_pred = model(x, t_tensor)
            
            # Parámetros para este paso t
            beta_t = scheduler.betas[t]
            alpha_t = scheduler.alphas[t]
            alpha_cumprod_t = scheduler.alphas_cumprod[t]
            
            # Fórmula para la media de x_{t-1} (derivada de la teoría)
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            mean = coef1 * (x - coef2 * noise_pred)
            
            if t > 0:
                # Añadimos ruido (excepto en el último paso)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        
        return x.cpu()

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scheduler = generador_difusion(num_timesteps=200)
model = eliminar_ruido(num_timesteps=200).to(device)
model.load_state_dict(torch.load("diffusion_1d.pth", map_location=device))

# Generar muestras
samples = sample(model, scheduler, n_samples=5000, device=device)

# Visualizar comparación
plt.figure(figsize=(10, 5))
plt.hist(samples.numpy(), bins=80, density=True, alpha=0.7, label='Generado')
# Datos reales para comparar
x_real = obtener_distribución_objetivo_1d(5000, "mixture").cpu().numpy()
plt.hist(x_real, bins=80, density=True, alpha=0.5, label='Real')
plt.legend()
plt.title("Comparación: distribución real vs generada")
plt.show()