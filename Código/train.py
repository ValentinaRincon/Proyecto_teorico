import torch
import torch.nn as nn
from generador_difusion import generador_difusion
from eliminar_ruido import eliminar_ruido
from data_1d import obtener_distribución_objetivo_1d

"""
El entrenamiento consiste en:

1-Coger datos limpios.
2-Elegir un t aleatorio.
3-Generar ruido.
4-Aplicar forward para obtener x_t.
5-Pedir a la red que prediga el ruido.
6-Comparar con el ruido real y ajustar pesos.
"""

# Hiperparámetros (valores que podemos ajustar)
batch_size = 1024
num_timesteps = 200
epochs = 20          # número de veces que recorremos los datos
lr = 1e-3            # learning rate (tasa de aprendizaje)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear generador y modelo
generador = generador_difusion(num_timesteps=num_timesteps)
model = eliminar_ruido(num_timesteps=num_timesteps).to(device)
optimizador = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()  # error cuadrático medio

print("Empezamos entrenamiento...")
for epoch in range(epochs):
    total_loss = 0
    # Hacemos 100 iteraciones por época (podríamos hacer más)
    for step in range(100):
        # 1. Generar batch de datos limpios
        x_0 = obtener_distribución_objetivo_1d(batch_size, "mixture").to(device)
        
        # 2. Muestrear t aleatorio para cada muestra
        t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
        
        # 3. Generar ruido
        noise = torch.randn_like(x_0)
        
        # 4. Obtener x_t usando forward
        x_t = generador.q_sample(x_0, t, noise)
        
        # 5. Predecir el ruido con el modelo
        noise_pred = model(x_t, t)
        
        # 6. Calcular pérdida (comparar ruido real con predicho)
        loss = loss_fn(noise_pred, noise)
        
        # 7. Retropropagación y optimización
        optimizador.zero_grad()  # limpia gradientes anteriores
        loss.backward()        # calcula gradientes
        optimizador.step()       # actualiza pesos
        
        total_loss += loss.item() # Acumulamos pérdida para mostrar promedio cada cierto tiempo
    
    print(f"Época {epoch+1}/{epochs}, Pérdida promedio: {total_loss/100:.6f}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "diffusion_1d.pth")
print("Modelo guardado.")