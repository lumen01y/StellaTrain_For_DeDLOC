import torch
import heapq

class TopKCompressor:
    def __init__(self, k, use_heap=False):
        self.k = k
        self.use_heap = use_heap   #True이면 heapq 사용, False이면 torch.topk() 사용

    def compress(self, grad):
        if self.use_heap:
            #힙정렬(indices를 리스트로 변환)
            top_k = heapq.nlargest(self.k, enumerate(grad.abs().tolist()), key=lambda x: x[1])
            indices, values = zip(*top_k)  #(index, value) 튜플 리스트를 분리
            indices = torch.tensor(indices, dtype=torch.long, device=grad.device)  #리스트 변환 후 텐서로 변환
            values = torch.tensor(values, dtype=grad.dtype, device=grad.device) #값도 텐서로 변환
        else:
            #기본 Top-K
            indices = torch.topk(grad.abs(), self.k, sorted=False).indices
            values = grad[indices]
        return indices, values

    def decompress(self, indices, values, shape):
        decompressed = torch.zeros(shape, device=values.device)
        decompressed[indices] = values
        return decompressed