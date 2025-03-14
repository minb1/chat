# chatRAG/model/base_handler.py
from abc import ABC, abstractmethod

class BaseModelHandler(ABC):
    @abstractmethod
    def generate_text(self, content: str) -> str:
        pass
