import torch
import threading

class SGDOptimizer(torch.nn.Module):
    def __init__(self, lr=0.01, momentum=0, weight_decay=0, dampening=0, 
                 nesterov=False, maximize=False, smart_momentum=False, device="cuda"):
        super(SGDOptimizer, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
        self.smart_momentum = smart_momentum
        self.iteration = 0  # m_iter
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.lock = threading.Lock()  # C++ mutex 대체

        # 상태 저장 (momentum buffer, last update)
        self.momentum_buffer = {}
        self.last_update = {}

    @torch.jit.ignore  # 🔥 JIT에서 lock을 무시하도록 처리
    def acquire_lock(self):
        """멀티스레드 보호를 위한 락 획득"""
        self.lock.acquire()

    @torch.jit.ignore  # 🔥 JIT에서 lock을 무시하도록 처리
    def release_lock(self):
        """멀티스레드 보호를 위한 락 해제"""
        self.lock.release()

    def optimize(self, param: torch.Tensor, name: str, grad_values: torch.Tensor, grad_indices: torch.Tensor):
        """
        PyTorch JIT 적용 - 실행 속도 최적화
        """
        param = param.to(self.device)  # GPU로 이동
        grad_values = grad_values.to(self.device)
        grad_indices = grad_indices.to(self.device)

        self.acquire_lock()  # 🔥 락 획득

        if name not in self.momentum_buffer:
            self.momentum_buffer[name] = torch.zeros_like(param, device=self.device)
            self.last_update[name] = torch.zeros_like(param, dtype=torch.int32, device=self.device)

        # Extract state
        momentum_buf = self.momentum_buffer[name]
        last_update = self.last_update[name]

        # Learning rate 처리
        lr = -self.lr if self.maximize else self.lr

        # 인덱스를 long 타입으로 변환
        grad_indices = grad_indices.long()

        # Gradient 업데이트
        for i in range(len(grad_values)):
            idx = grad_indices[i].item()
            grad = grad_values[i].item()

            # Weight Decay 적용
            if self.weight_decay != 0:
                grad += self.weight_decay * param[idx].item()

            # Momentum 계산
            if self.momentum != 0:
                if self.smart_momentum:
                    momentum_factor = self.momentum if self.iteration == 0 else self.momentum ** (self.iteration - last_update[idx].item())
                else:
                    momentum_factor = self.momentum

                if not (0 <= momentum_factor < 1):
                    raise ValueError(f"Invalid momentum {momentum_factor}")

                if self.iteration == 0:
                    momentum_buf[idx] = grad
                else:
                    momentum_buf[idx] = momentum_factor * momentum_buf[idx] + (1 - self.dampening) * grad

                if self.nesterov:
                    grad += momentum_factor * momentum_buf[idx]
                else:
                    grad = momentum_buf[idx]

            # 파라미터 업데이트
            param[idx] -= lr * grad

            # Smart Momentum을 위한 업데이트 타이밍 저장
            if self.smart_momentum:
                last_update[idx] = self.iteration

        self.iteration += 1
        self.release_lock()  # 🔥 락 해제

    def configure(self, option_name, option_value):
        """
        Optimizer 하이퍼파라미터 설정
        """
        if option_name == "lr":
            self.lr = option_value
        elif option_name == "momentum":
            self.momentum = option_value
        elif option_name == "weight_decay":
            self.weight_decay = option_value
        elif option_name == "dampening":
            self.dampening = option_value
        elif option_name == "nesterov":
            self.nesterov = option_value
        elif option_name == "maximize":
            self.maximize = option_value
        elif option_name == "smart_momentum":
            self.smart_momentum = option_value
        else:
            raise ValueError(f"Unknown option: {option_name}")

# `torch.compile()`을 적용하여 PyTorch 내부 최적화 활성화
SGDOptimizer = torch.compile(SGDOptimizer)
