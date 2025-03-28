
from typing import Iterable, Tuple, Union
import torch

def gather(lst: Iterable[torch.Tensor]) -> torch.Tensor: ...

def configure(master_addr: str, master_port: int, world_size: int, 
              rank: int, local_session_id: int, local_world_size: int = 0, local_rank: int = 0, method: str = 'thresholdv', gradient_compression: int = 1) -> None: ...

def set_optimizer(optimizer_name: str) -> None: ...
def configure_compression(method: str) -> None: ...
def configure_optimizer(option_name: str, option_val: Union[float, bool]) -> None: ...
def configure_compression_ratio(ratio: float) -> None: ...

def gradient_accumulation() -> int: ...
def is_debug_accuracy_mode()-> bool: ...

def barrier() -> None: ...
def synchronize() -> None: ...

def pre_train_init(self, layer_idx: int, name: str, param_tensor: torch.Tensor) -> None: ...
def post_backward_process(self, layer_idx: int, name: str, grad_tensor: torch.Tensor, param_tensor: torch.Tensor) -> None: ...
def pre_forward_process(self, layer_idx: int, name: str) -> None: ...

def force_model_sync(layer_idx: int, name: str) -> None: ...

def compress(grad: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]: ...

class SGD():    
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False): ...
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...