import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
from tqdm import tqdm


def load_data(base_path="./data", device="cpu"):
    """Load the data in PyTorch Tensor."""
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0  # Fill missing entries with 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix).to(device)
    train_matrix = torch.FloatTensor(train_matrix).to(device)

    return zero_train_matrix, train_matrix, valid_data, test_data


class ContrastiveEncoder(nn.Module):
    def __init__(self, num_questions, latent_dim=100):
        """Initialize Contrastive Encoder."""
        super(ContrastiveEncoder, self).__init__()
        self.fc1 = nn.Linear(num_questions, 1024)
        self.fc2 = nn.Linear(1024, latent_dim)

    def forward(self, x):
        """Forward pass to generate embeddings."""
        h1 = F.relu(self.fc1(x))
        return F.normalize(self.fc2(h1), dim=-1)  # Normalize embeddings


def contrastive_loss(z1, z2, temperature=0.5):
    """
    Compute contrastive loss (NT-Xent) for a batch of embeddings.

    :param z1: Batch of embeddings (anchor).
    :param z2: Batch of embeddings (positive pair).
    :param temperature: Temperature scaling factor.
    :return: Contrastive loss.
    """
    batch_size = z1.shape[0]
    similarity_matrix = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(batch_size).to(z1.device)
    return F.cross_entropy(similarity_matrix, labels)


def pretrain_contrastive_encoder(encoder, data, num_epochs, lr, temperature=0.5):
    """
    Pre-train the encoder using contrastive learning.

    :param encoder: ContrastiveEncoder model.
    :param data: Training data (torch.Tensor).
    :param num_epochs: Number of training epochs.
    :param lr: Learning rate.
    :param temperature: Temperature for contrastive loss.
    """
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)  # Adjust weight_decay as needed
    encoder.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for user_id in range(data.shape[0]):
            inputs = data[user_id].unsqueeze(0)

            # Augment data for positive pair
            positive_pair = inputs + 0.01 * torch.randn_like(inputs)  # Adjusted augmentation intensity

            # Normalize inputs and positive pair
            z1 = encoder(inputs)
            z2 = encoder(positive_pair)

            # Debug: Print norms to ensure embeddings are meaningful
            if epoch == 0 and user_id == 0:
                tqdm.write(f"z1 norm: {torch.norm(z1).item()}, z2 norm: {torch.norm(z2).item()}")

            # Compute contrastive loss
            loss = contrastive_loss(z1, z2, temperature)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


class ContrastiveAutoEncoder(nn.Module):
    def __init__(self, encoder, num_questions, latent_dim=100):
        """Initialize the AutoEncoder with a pre-trained encoder."""
        super(ContrastiveAutoEncoder, self).__init__()
        self.encoder = encoder  # Pre-trained encoder

        # Decoder: Maps latent space back to input space
        self.fc2 = nn.Linear(latent_dim, 2048)
        self.fc3 = nn.Linear(2048, num_questions)

    def forward(self, x):
        """Forward pass."""
        z = self.encoder(x)
        h2 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h2))


def train_contrastive_autoencoder(model, lr, train_data, zero_train_data, valid_data, num_epoch, device="cpu"):
    """Train the autoencoder after pre-training the encoder."""
    tqdm.write("Training autoencoder")
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Adjust weight_decay as needed

    num_students = train_data.shape[0]

    for epoch in tqdm(range(num_epoch)):
        train_loss = 0.0

        for user_id in range(num_students):
            inputs = zero_train_data[user_id].unsqueeze(0).to(device)
            target = inputs.clone()

            optimizer.zero_grad()
            reconstruction = model(inputs)

            # Mask the target for valid entries
            nan_mask = torch.isnan(train_data[user_id].unsqueeze(0)).to(device)
            target[nan_mask] = reconstruction[nan_mask]

            loss = F.mse_loss(reconstruction, target, reduction="sum")
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, device)
        tqdm.write(f"Epoch: {epoch + 1}/{num_epoch}, Training Loss: {train_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")


def evaluate(model, train_data, valid_data, device="cpu"):
    """Evaluate the valid_data on the current model."""
    model.eval()
    model.to(device)

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = train_data[u].unsqueeze(0).to(device)
        reconstruction = model(inputs)
        guess = reconstruction[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")

    zero_train_matrix, train_matrix, valid_data, test_data = load_data("./data", device)

    latent_dim = 200

    # Step 1: Pre-train encoder with contrastive learning
    encoder = ContrastiveEncoder(train_matrix.shape[1], latent_dim=latent_dim).to(device)
    pretrain_contrastive_encoder(encoder, zero_train_matrix, num_epochs=10, lr=0.001)

    # Step 2: Train autoencoder with pre-trained encoder
    model = ContrastiveAutoEncoder(encoder, train_matrix.shape[1], latent_dim=latent_dim).to(device)
    train_contrastive_autoencoder(
        model, lr=5e-5, train_data=train_matrix, zero_train_data=zero_train_matrix, valid_data=valid_data, num_epoch=15, device=device
    )

    # Step 3: Evaluate on the test set
    test_acc = evaluate(model, zero_train_matrix, test_data, device)
    tqdm.write(f"Test Accuracy: {test_acc:.4f}")

    # save the model
    # torch.save(model.state_dict(), "contrastive_autoencoder.pth")


if __name__ == "__main__":
    main()