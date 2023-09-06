import torch
import torch.nn as nn

from typing import List, Union

from transformers import AutoModel, AutoTokenizer

class BaseEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pooler = kwargs.get("pooler", "cls")
        self.max_seq_len = kwargs.get("max_seq_len", 128)
        self.initialize_model(model_name_or_path, **kwargs)

    def initialize_model(self, model_name_or_path: str, **kwargs) -> None:
        """
        Initialize the model and tokenizer
        :param model_name_or_path: the name of the model or the path to the model
        :return: None
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        self.model.to(self.device)

    def encode(self, input: Union[str, List[str]]):
        """
        Encode a single string or a list of strings into embeddings
        :param input: a single string or a list of strings
        :return: a tensor of shape (batch_size, embedding_dim)
        """
        tokens = self.tokenizer(input, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False)
        with torch.no_grad():
            embedding = self(**tokens)
        
        return embedding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass of the model
        :param input_ids: a tensor of shape (batch_size, max_seq_len)
        :param attention_mask: a tensor of shape (batch_size, max_seq_len)
        :return: a tensor of shape (batch_size, embedding_dim)
        """
        output = self.model(input_ids, attention_mask=attention_mask)

        if self.pooler == "cls":
            return output.last_hidden_state[:, 0]
        elif self.pooler == "mean":
            return output.last_hidden_state.mean(dim=1)

    def load_from_path(self, path: str) -> None:
        """
        Load the model from a path
        :param path: the path to the model
        :return: None
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class QueryEncoder(BaseEncoder):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

class DocumentEncoder(BaseEncoder):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)