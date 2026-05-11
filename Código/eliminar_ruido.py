import torch
import torch.nn as nn

class eliminar_ruido(nn.Module):
    def __init__(self, hidden_dims=[128, 256, 128], num_timesteps=200):
        super().__init__()
        
        # Embedding del paso de tiempo: convierte un entero (t) en un vector de tamaño hidden_dims[0]
        self.time_embed = nn.Embedding(num_timesteps, hidden_dims[0]) # Crea una capa de embedding que toma un entero (el paso de tiempo t) y lo convierte en un vector de tamaño hidden_dims[0]
        
        # Construimos las capas de la red
        layers = []
        input_dim = 1 + hidden_dims[0]  # concatenamos el valor x (1) con el embedding de t
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())  # función de activación
            input_dim = h
        # Capa final: produce un solo número (el ruido predicho)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_t, t):
        # x_t: (batch, 1)
        # t: (batch,) enteros
        t_emb = self.time_embed(t)  # (batch, hidden_dims[0])
        # Concatenamos a lo largo de la dimensión 1 (la de características)
        x_t = x_t.view(-1, 1)  # por si acaso
        h = torch.cat([x_t, t_emb], dim=1)  # (batch, 1 + hidden_dims[0]) # Juntamos el valor de x con el vector de t. La red tendrá acceso a ambos para decidir cuánto ruido hay que predecir.
        # Pasamos por la red
        noise_pred = self.net(h)  # (batch, 1)
        return noise_pred