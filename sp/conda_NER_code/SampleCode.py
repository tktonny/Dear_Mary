import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Tuple, Dict, Optional, Any

from .NER_dataloader import create_sequence_dataloaders

# A sentence is a list of (word, tag) tuples.
# For example, [("hello", "O"), ("world", "O"), ("!", "O")]
Sentence = List[Tuple[str, str]]

def get_device() -> torch.device:
    """
    Use GPU when it is available; use CPU otherwise.
    See https://pytorch.org/docs/stable/notes/cuda.html#device-agnostic-code
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LSTM(nn.Module):
    """
    Long short-term memory for NER
    """

    def __init__(self, vocab: Dict[str, Vocab], d_emb: int, d_hidden: int, bidirectional: bool) -> None:
        """
        Initialize an LSTM
        Parameters:
            `vocab`: vocabulary of words and tags
            `d_emb`: dimension of word embeddings (D)
            `d_hidden`: dimension of the hidden layer (H)
        """
        super().__init__()
        # TODO: Create the word embeddings (nn.Embedding),
        #       the LSTM (nn.LSTM) and the output layer (nn.Linear).
        # START HERE
        raise NotImplementedError
        # END

    def forward(
        self, word_idxs: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Given words in sentences, predict the logits of the NER tag.
        Parameters:
            `word_idxs`: a batch_size x max_len tensor
            `valid_mask`: a batch_size x max_len tensor
        Return values:
            `logits`: a batch_size x max_len x 5 tensor
        """
        # TODO: Implement the forward pass
        # Hint: You may need to use nn.utils.rnn.pack_padded_sequence and nn.utils.rnn.pad_packed_sequence
        #       to handle sentences with different lengths.
        # START HERE
        raise NotImplementedError
        # END
        return logits
    
    
def eval_metrics(ground_truth: List[int], predictions: List[int]) -> Dict[str, Any]:
    """
    Calculate various evaluation metrics such as accuracy and F1 score
    Parameters:
        `ground_truth`: the list of ground truth NER tags
        `predictions`: the list of predicted NER tags
    """
    f1_scores = f1_score(ground_truth, predictions, average=None)
    return {
        "accuracy": accuracy_score(ground_truth, predictions),
        "f1": f1_scores,
        "average f1": np.mean(f1_scores),
        "confusion matrix": confusion_matrix(ground_truth, predictions),
    }

def train_lstm(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    silent: bool = False,  # whether to print the training loss
) -> Tuple[float, Dict[str, Any]]:
    """
    Train the LSTM model.
    Return values:
        1. the average training loss
        2. training metrics such as accuracy and F1 score
    """
    model.train()
    ground_truth = []
    predictions = []
    losses = []
    report_interval = 100

    for i, data_batch in enumerate(loader):
        word_idxs = data_batch["word_idxs"].to(device, non_blocking=True)
        tag_idxs = data_batch["tag_idxs"].to(device, non_blocking=True)
        valid_mask = data_batch["valid_mask"].to(device, non_blocking=True)
 
        # 1. Perform the forward passs to calculate the model's output. Save it to the variable "logits".
        # 2. Calculate the loss using the output and the ground truth tags. Save it to the variable "loss".
        # 3. Perform the backward pass to calculate the gradient.
        # 4. Use the optimizer to update model parameters.

        logits = model.forward(word_idxs, valid_mask)
        loss = F.cross_entropy(logits[valid_mask], tag_idxs[valid_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits_flat = logits.view(-1, 5)
        tag_idxs_flat = tag_idxs.view(-1, 1)
        valid_mask_flat = valid_mask.view(-1, 1)

        losses.append(loss.item())
        ground_truth.extend(tag_idxs_flat[valid_mask_flat].tolist())
        predictions.extend(logits_flat[valid_mask_flat.squeeze(-1)].argmax(dim=-1).tolist())

        if not silent and i > 0 and i % report_interval == 0:
            print(
                "\t[%06d/%06d] Loss: %f"
                % (i, len(loader), np.mean(losses[-report_interval:]))
            )

    return np.mean(losses), eval_metrics(ground_truth, predictions)


def validate_lstm(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, Dict[str, Any]]:
    """
    Validate the model.
    Return the validation loss and metrics.
    """
    model.eval()
    ground_truth = []
    predictions = []
    losses = []

    with torch.no_grad():

        for data_batch in loader:
            word_idxs = data_batch["word_idxs"].to(device, non_blocking=True)
            tag_idxs = data_batch["tag_idxs"].to(device, non_blocking=True)
            valid_mask = data_batch["valid_mask"].to(device, non_blocking=True)

            logits = model.forward(word_idxs, valid_mask)
            loss = F.cross_entropy(logits[valid_mask], tag_idxs[valid_mask])

            logits_flat = logits.view(-1, 5)
            tag_idxs_flat = tag_idxs.view(-1, 1)
            valid_mask_flat = valid_mask.view(-1, 1)

            losses.append(loss.item())
            ground_truth.extend(tag_idxs_flat[valid_mask_flat].tolist())
            predictions.extend(logits_flat[valid_mask_flat.squeeze(-1)].argmax(dim=-1).tolist())

    return np.mean(losses), eval_metrics(ground_truth, predictions)


def train_val_loop_lstm(hyperparams: Dict[str, Any]) -> None:
    """
    Train and validate the LSTM model for a number of epochs.
    """
    print("Hyperparameters:", hyperparams)
    # Create the dataloaders
    loader_train, loader_val, vocab = create_sequence_dataloaders(
        hyperparams["batch_size"]
    )
    # Create the model
    model = LSTM(
        vocab,
        hyperparams["d_emb"],
        hyperparams["d_hidden"],
        hyperparams["bidirectional"],
    )
    device = get_device()
    model.to(device)
    print(model)
    # Create the optimizer
    optimizer = optim.RMSprop(
        model.parameters(), hyperparams["learning_rate"], weight_decay=hyperparams["l2"]
    )

    # Train and validate
    for i in range(hyperparams["num_epochs"]):
        print("Epoch #%d" % i)

        print("Training..")
        loss_train, metrics_train = train_lstm(model, loader_train, optimizer, device)
        print("Training loss: ", loss_train)
        print("Training metrics:")
        for k, v in metrics_train.items():
            print("\t", k, ": ", v)

        print("Validating..")
        loss_val, metrics_val = validate_lstm(model, loader_val, device)
        print("Validation loss: ", loss_val)
        print("Validation metrics:")
        for k, v in metrics_val.items():
            print("\t", k, ": ", v)

    print("Done!")


if __name__ == '__main__':
    train_val_loop_lstm({
        "bidirectional": True,
        "batch_size": 512,
        "d_emb": 64,
        "d_hidden": 128,
        "num_epochs": 15,
        "learning_rate": 0.005,
        "l2": 1e-6,
    })