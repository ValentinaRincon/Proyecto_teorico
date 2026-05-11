import torch

"""
Este archivo generará números que siguen una distribución específica. 
Por ejemplo, una mezcla de dos campanas (gaussianas) centradas en -2 y 2.
"""

def obtener_distribución_objetivo_1d(batch_size, distribution_type="mixture"):

    if distribution_type == "mixture":

        # Elegimos aleatoriamente para cada muestra si viene de la primera o segunda campana
        means = torch.tensor([-2.0, 2.0])
        stds = torch.tensor([0.5, 0.5])

        # Asignamos a cada muestra un índice 0 o 1
        components = torch.randint(0, len(means), (batch_size,)) #Crea un tensor de enteros aleatorios entre 0 y 1 (ya que len(means)=2) de tamaño batch_size.

        # Generamos números aleatorios con la media y desviación correspondiente
        x = torch.normal(means[components], stds[components]) #Genera números aleatorios con media means[components] y desviación stds[components]

        return x.view(-1, 1)  # devolvemos con forma (batch, 1)
    
    elif distribution_type == "uniform":

        x = torch.rand(batch_size, 1) * 6 - 3  # uniforme entre -3 y 3
        return x
    
    else:

        # Por defecto, una normal estándar
        x = torch.randn(batch_size, 1)
        return x
    