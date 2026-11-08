
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math

# 1. Chargement de ResNet50 pré-entraîné
resnet_pretrained = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# 2. Suppression des deux dernières couches
# On conserve la carte de caractéristiques spatiales
cnn_features = nn.Sequential(
    *list(resnet_pretrained.children())[:-2]
)

# 3. Gel des poids du ResNet
for param in cnn_features.parameters():
    param.requires_grad = False


# 4. Vérification des dimensions
if __name__ == "__main__":
    cnn_features.eval()

    images = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        features = cnn_features(images)

    print("Dimensions des images :", images.shape)
    print("Dimensions des caractéristiques :", features.shape)

    print(
        "Paramètres entraînables :",
        sum(p.numel() for p in cnn_features.parameters()
            if p.requires_grad)
    )
class AttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionModule, self).__init__()

        # Couche pour calculer le score d'attention
        self.attention_layer = nn.Conv2d(
            feature_dim + hidden_dim, 1, 1
        )

    def forward(self, features, hidden_state):
        # features : (batch_size, 2048, 7, 7)
        # hidden_state : (batch_size, hidden_dim)

        # Étendre l'état caché sur les 49 régions
        hidden_state_expanded = hidden_state.unsqueeze(2).unsqueeze(3).repeat(
            1, 1, features.size(2), features.size(3)
        )

        # Concaténer les caractéristiques et l'état caché
        concat_features = torch.cat(
            (features, hidden_state_expanded), dim=1
        )

        # Calculer les scores d'attention
        attention_scores = self.attention_layer(
            concat_features
        ).squeeze(1)

        # Softmax sur les 49 régions
        attention_weights = F.softmax(
            attention_scores.view(attention_scores.size(0), -1),
            dim=1
        )

        attention_weights = attention_weights.view(
            attention_scores.size(0),
            1,
            attention_scores.size(1),
            attention_scores.size(2)
        )

        # Somme pondérée des caractéristiques visuelles
        context_vector = torch.sum(
            torch.sum(features * attention_weights, dim=3),
            dim=2
        )

        return context_vector, attention_weights
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size,
                 attention_module, attention_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_module = attention_module

        # Mot + état caché précédent + contexte visuel
        total_dim = input_size + hidden_size + attention_size

        self.lin_i = nn.Linear(total_dim, hidden_size)
        self.lin_f = nn.Linear(total_dim, hidden_size)
        self.lin_c = nn.Linear(total_dim, hidden_size)
        self.lin_o = nn.Linear(total_dim, hidden_size)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, features, init_states=None):
        # x : (batch_size, sequence_size, input_size)

        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(
                bs, self.hidden_size
            ).to(x.device)

            c_t = torch.zeros(
                bs, self.hidden_size
            ).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            # Calcul de l'attention
            att_t, _ = self.attention_module(
                features, h_t
            )

            # Concaténation du contexte, du mot et de l'état caché
            concat = torch.cat(
                (att_t, x_t, h_t), dim=1
            )

            # Portes du LSTM
            i_t = torch.sigmoid(self.lin_i(concat))
            f_t = torch.sigmoid(self.lin_f(concat))
            g_t = torch.tanh(self.lin_c(concat))
            o_t = torch.sigmoid(self.lin_o(concat))

            # Mise à jour de la mémoire
            c_t = f_t * c_t + i_t * g_t

            # Mise à jour de l'état caché
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)

        return hidden_seq, (h_t, c_t)

class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, embedding_model, lstm_hidden_dim):
        super(ImageCaptioningModel, self).__init__()

        # Extraction des caractéristiques de ResNet50
        self.cnn = nn.Sequential(
            *list(cnn_model.children())[:-2]
        )

        # Gel des poids de ResNet50
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Module d'attention
        self.attention = AttentionModule(
            feature_dim=2048,
            hidden_dim=lstm_hidden_dim
        )

        # Embeddings Word2Vec
        self.embeddings = embedding_model

        vocab_size = embedding_model.weight.shape[0]
        emb_dim = embedding_model.weight.shape[1]

        # LSTM avec attention
        self.lstm = LSTMWithAttention(
            emb_dim,
            lstm_hidden_dim,
            self.attention,
            2048
        )

        # Prédiction du prochain mot
        self.fc_out = nn.Linear(
            lstm_hidden_dim,
            vocab_size
        )

    def forward(self, images, captions):
        # Extraction des caractéristiques visuelles
        with torch.no_grad():
            cnn_features = self.cnn(images)

        # Conversion des mots en embeddings
        emb = self.embeddings(captions)

        # LSTM avec attention
        lstm_outputs, (hidden_state, _) = self.lstm(
            emb, cnn_features
        )

        # Prédiction des mots
        output = self.fc_out(lstm_outputs)

        return output
