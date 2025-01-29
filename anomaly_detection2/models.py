import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = "Autoencoder"


#Attention-Autoencoder-TCN 
class AttentionAutoencoderTCN(Autoencoder):
    def __init__(self, input_channels):
        super(AttentionAutoencoderTCN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding='same'),  
            nn.ReLU()
        )

        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, input_channels, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )
        self.info = "AttentionAutoencoderTCN"
    def forward(self, x):

        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        
        encoded = encoded.permute(0, 2, 1)
        # Attention mechanism
        attn_output, _ = self.attention(encoded, encoded, encoded)

        attn_output = attn_output.permute(0, 2, 1)
        decoded = self.decoder(attn_output)
        return decoded.permute(0, 2, 1)

#CNN-Autoencoder
class TemporalCNNAutoencoder(Autoencoder):
    def __init__(self, input_dim, cnn_channels = 32):
        super(TemporalCNNAutoencoder, self).__init__()
        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=cnn_channels, out_channels=input_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.info = "TemporalCNNAutoencoder"

    def forward(self, x):
        # Change to (batch, features, timesteps) for CNN
        x = x.permute(0, 2, 1)
        encoded = self.encoder_cnn(x)
        decoded = self.decoder_cnn(encoded)
        # Return to original shape (batch, timesteps, features)
        return decoded.permute(0, 2, 1)
    
#LSTM and CNN Autoencoder
class LSTMCNNAutoencoder(Autoencoder):
    def __init__(self, input_dim, hidden_dim = 64, cnn_channels = 32):
        super(LSTMCNNAutoencoder, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_cnn = nn.Conv1d(in_channels=hidden_dim, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.decoder_cnn = nn.ConvTranspose1d(in_channels=cnn_channels, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.info = "LSTMCNNAutoencoder"
    def forward(self, x):
        # LSTM Encoder
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden.permute(1, 2, 0)  # Reshape for CNN (batch, hidden_dim, timesteps)

        # CNN Encoder
        encoded_cnn = self.encoder_cnn(hidden)

        # CNN Decoder
        decoded_cnn = self.decoder_cnn(encoded_cnn)
        decoded_cnn = decoded_cnn.permute(2, 0, 1)  # Reshape back for LSTM (timesteps, batch, hidden_dim)

        # LSTM Decoder
        decoded_lstm, _ = self.decoder_lstm(decoded_cnn.permute(1, 0, 2))
        return decoded_lstm


#LSTM Autoencoder
class LSTMAutoencoder(Autoencoder):
    def __init__(self, input_dim, hidden_dim= 64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.info = "LSTMAutoencoder"
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # Repeat for each timestep
        reconstructed, _ = self.decoder(hidden)
        return reconstructed