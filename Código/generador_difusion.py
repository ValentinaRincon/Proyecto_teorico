import torch

class generador_difusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Plan de ruido: los betas controlan cuánto ruido se añade en cada paso
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Derivamos los alphas (1 - beta)
        self.alphas = 1.0 - self.betas
        
        # Producto acumulado de alphas (esto es para la fórmula de un salto)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Raíces cuadradas necesarias para la fórmula
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None): #Método que aplica el proceso forward.

        """
        Toma x_0 y le añade ruido hasta el paso t.
        x_0: tensor de forma (batch_size, 1) (nuestros datos)
        t: tensor de enteros (batch_size,) (pasos para cada muestra)
        noise: si no se da, se genera aleatorio
        """

        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Obtenemos los factores escalares para cada elemento según su t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # Aplicamos la fórmula: x_t = factor1 * x_0 + factor2 * noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t