import numpy as np
import torch
import threading
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - INFO - %(message)s")

class ThresholdvCompressor:
    _global_threshold_map = {}  # ✅ 전역 캐시 유지
    _lock = threading.Lock()  # ✅ 전역 잠금

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        logging.info("ThresholdvCompressor initialized with use_gpu=%s", use_gpu)

    def compress(self, name, src, k, dst_idx, dst_val, idx_offset=0):
        logging.info("Starting compression for %s with k=%d", name, k)
        return self.impl_simd_gpu(name, src, k, dst_idx, dst_val, idx_offset)

    def impl_simd_gpu(self, name, src, k, dst_idx, dst_val, idx_offset=0):
        """GPU 기반 Threshold 압축"""
        device = "cuda" if self.use_gpu else "cpu"
        src_tensor = torch.tensor(src, dtype=torch.float32, device=device)
        abs_src = torch.abs(src_tensor)  # ✅ 항상 abs_src를 계산

        # ✅ 전역 threshold_map 활용하여 캐싱 유지
        with ThresholdvCompressor._lock:  # ✅ 전역 잠금 사용
            if name not in ThresholdvCompressor._global_threshold_map:
                logging.info("⚠️ Threshold not found in cache for '%s'. Computing new one.", name)
                threshold = np.percentile(abs_src.cpu().numpy(), (1 - k / len(abs_src)) * 100)
                ThresholdvCompressor._global_threshold_map[name] = threshold  # ✅ 전역 캐시에 저장
                logging.info("🆕 Calculated new threshold for '%s': %.6f", name, threshold)
            else:
                threshold = ThresholdvCompressor._global_threshold_map[name]
                logging.info("✅ Using cached threshold for '%s': %.6f", name, threshold)

        # ✅ 마스크 필터링을 통해 값 선택
        mask = abs_src >= threshold  # ✅ 이제 abs_src가 항상 존재함!
        valid_indices = torch.nonzero(mask, as_tuple=True)[0]
        valid_values = src_tensor[valid_indices]

        cnt_found = min(k, len(valid_indices))
        dst_idx[:cnt_found] = valid_indices[:cnt_found].cpu().numpy()
        dst_val[:cnt_found] = valid_values[:cnt_found].cpu().numpy()

        # ✅ AIMD 기반 임계값 조정 (Adaptive Increase/Multiplicative Decrease)
        if k > cnt_found:
            threshold *= 0.99  # 드롭을 줄임
        elif k < cnt_found:
            threshold += 0.01 * torch.max(abs_src).item()  # 더 많은 드롭 수행

        ThresholdvCompressor._global_threshold_map[name] = threshold
        logging.info("🔄 AIMD-adjusted threshold for '%s': %.6f", name, threshold)

        # ✅ 보조 버퍼 추가 (초과된 데이터를 저장하여 보완)
        remaining_k = k - cnt_found
        if remaining_k > 0:
            sorted_abs_src, sorted_idx = torch.sort(abs_src, descending=True)
            backup_indices = sorted_idx[:remaining_k].cpu().numpy()
            backup_values = src_tensor[backup_indices].cpu().numpy()
            
            dst_idx[cnt_found:cnt_found+remaining_k] = backup_indices
            dst_val[cnt_found:cnt_found+remaining_k] = backup_values
            cnt_found += remaining_k
            logging.info("🛠 Added backup elements to maintain k=%d", k)

        logging.info("Compression complete for '%s'. Found %d elements.", name, cnt_found)
        return cnt_found

    def decompress(self, compressed_idx: np.ndarray, compressed_val: np.ndarray, original_size: int):
        """✅ GPU 친화적인 압축 해제 함수"""
        device = "cuda" if self.use_gpu else "cpu"
        decompressed = torch.zeros(original_size, dtype=torch.float32, device=device)

        # ✅ NumPy → Torch 변환 후 할당
        decompressed[compressed_idx] = torch.tensor(compressed_val, dtype=torch.float32, device=device)

        return decompressed
