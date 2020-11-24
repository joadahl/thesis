from tensorflow import keras
from encoder import encoder
from decoder import decoder

class vae(keras.Model):
    def __init__(self, latent_dim):
        super(vae, self).__init__()
        self.encoder = encoder(latent_dim)
        self.decoder = decoder()

