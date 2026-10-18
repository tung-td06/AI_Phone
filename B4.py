"""
Module B4: Fine-tune Embedding Model cho bài toán Tìm kiếm & Truy vấn thông số điện thoại (Phone Specs Retrieval).

Quy trình:
1. Nạp mô hình nền (Base Model): sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   - Hỗ trợ đa ngôn ngữ (tiếng Việt, tiếng Anh, 50+ ngôn ngữ).
   - Kích thước nhẹ (~118M tham số, embedding dim 384), tốc độ suy luận nhanh (~10-15ms/query).
2. Tối ưu hóa phần cứng GPU NVIDIA RTX 3060 (12GB VRAM):
   - Chạy FP16 Mixed Precision giúp giảm 50% VRAM và tăng tốc huấn luyện 2-3 lần.
   - Batch size tối ưu: 32 đến 48 (kèm gradient accumulation nếu cần).
3. Sử dụng Loss: MultipleNegativesRankingLoss (MNRL)
   - Tận dụng bộ ba Triplets (Anchor, Positive, Hard Negative) được sinh từ B3:
     + Hard Negative 1: Cùng máy nhưng khác chuyên mục (chống nhầm lẫn tính năng).
     + Hard Negative 2: Cùng chuyên mục nhưng khác model (chống nhầm lẫn đời máy).
   - Trong mỗi batch, mỗi Anchor vừa được so sánh với Hard Negative trực tiếp, vừa coi tất cả các
     Positive/Negative khác trong batch là in-batch negatives (tạo ra hàng chục tín hiệu tương phản mỗi bước).
4. Đánh giá tự động (TripletEvaluator) trên tập Validation định kỳ (eval_steps).
5. Lưu mô hình tốt nhất (Best Checkpoint) và xuất mô hình hoàn thiện cuối cùng.
6. Chạy Benchmark so sánh trước & sau Fine-tune (Before vs After) với các test case thực tế.
"""

import os
import sys
import json
import time
import math
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# Đảm bảo terminal Windows hiển thị tiếng Việt UTF-8 chuẩn xác
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Bỏ qua các cảnh báo deprecation không ảnh hưởng logic
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
try:
    from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
except ImportError:
    import importlib
    _eval_mod = importlib.import_module("sentence_transformers.evaluation")
    TripletEvaluator = getattr(_eval_mod, "TripletEvaluator")
from sentence_transformers.util import cos_sim

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ==============================================================================
# HẰNG SỐ CẤU HÌNH MẶC ĐỊNH
# ==============================================================================
DEFAULT_BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATA_DIR = Path(r"p:\train AI\dataset_finetune")
DEFAULT_OUTPUT_DIR = Path(r"p:\train AI\final_phone_embedding_model")
DEFAULT_CHECKPOINT_DIR = Path(r"p:\train AI\checkpoints_phone_embedding")
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.1


# ==============================================================================
# 1. KIỂM TRA PHẦN CỨNG & THIẾT BỊ GPU
# ==============================================================================
def check_hardware_and_environment() -> Dict[str, Any]:
    """
    Kiểm tra tình trạng phần cứng máy tính:
    - GPU NVIDIA, hỗ trợ CUDA, dung lượng VRAM.
    - Cảnh báo và hướng dẫn cài đặt PyTorch CUDA nếu đang chạy trên CPU.
    """
    cuda_available = torch.cuda.is_available()
    device_name = "cpu"
    vram_gb = 0.0
    gpu_model = "CPU Only"
    supports_fp16 = False

    if cuda_available:
        device_name = "cuda"
        gpu_model = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        supports_fp16 = True
    else:
        # Kiểm tra xem có nvidia-smi trên máy Windows không
        import shutil
        has_nvidia_smi = shutil.which("nvidia-smi") is not None
        if has_nvidia_smi:
            gpu_model = "NVIDIA GPU phát hiện (nhưng PyTorch chưa kích hoạt CUDA)"

    info = {
        "cuda_available": cuda_available,
        "device": device_name,
        "gpu_model": gpu_model,
        "vram_gb": vram_gb,
        "supports_fp16": supports_fp16,
    }
    return info


def print_hardware_summary(hw: Dict[str, Any], batch_size: int, fp16_requested: bool):
    """In bảng trạng thái phần cứng và đề xuất cấu hình huấn luyện."""
    print("-" * 75)
    print(" [1/5] THÔNG TIN PHẦN CỨNG & MÔI TRƯỜNG HUẤN LUYỆN:")
    print(f" - Thiết bị thực thi       : {hw['device'].upper()}")
    print(f" - Tên GPU / Phần cứng     : {hw['gpu_model']}")

    if hw["cuda_available"]:
        print(f" - Bộ nhớ VRAM khả dụng    : {hw['vram_gb']:.2f} GB")
        print(f" - Khả năng tăng tốc FP16  : {'CÓ (Khả dụng)' if hw['supports_fp16'] else 'KHÔNG'}")
        if "3060" in hw["gpu_model"]:
            print(" -> [KHUYÊN DÙNG RTX 3060] VRAM 12GB hoàn hảo cho Batch Size 32-48 với FP16!")
    else:
        print(" ! CẢNH BÁO: PyTorch đang chạy ở chế độ CPU.")
        if "NVIDIA" in hw["gpu_model"]:
            print("   -> Bạn có GPU NVIDIA nhưng PyTorch hiện tại là bản CPU-only.")
            print("   -> Để kích hoạt GPU RTX 3060 chạy nhanh hơn gấp 10-20 lần, hãy cài đặt:")
            print("      pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124")
        print("   -> Chế độ FP16 sẽ tạm thời tắt khi chạy trên CPU.")
    print("-" * 75)


# ==============================================================================
# 2. NẠP DỮ LIỆU HUẤN LUYỆN TỪ JSONL
# ==============================================================================
def load_dataset_from_jsonl(
    file_path: Path,
    data_type: str = "triplets",
    limit: Optional[int] = None,
) -> Dataset:
    """
    Nạp dữ liệu từ file .jsonl sử dụng Apache Arrow Memory Mapping (tiết kiệm 95% RAM).
    Thay vì nạp hàng trăm nghìn chuỗi vào Python list gây tràn 99% RAM,
    Arrow mmap ánh xạ trực tiếp từ đĩa và nạp từng batch vào GPU khi cần.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path.resolve()}")

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f" -> Đang nạp {file_path.name} ({file_size_mb:.1f} MB) qua Arrow Memory Mapping...")

    t0 = time.time()
    from datasets import load_dataset

    # Nạp qua memory-mapped Arrow
    ds = load_dataset("json", data_files=str(file_path.resolve()), split="train")

    if "query" in ds.column_names and "anchor" not in ds.column_names:
        ds = ds.rename_column("query", "anchor")

    # Giữ lại đúng các cột huấn luyện cần thiết, loại bỏ metadata thừa để giải phóng RAM
    if data_type == "triplets":
        cols_to_keep = [c for c in ["anchor", "positive", "negative"] if c in ds.column_names]
    else:
        cols_to_keep = [c for c in ["anchor", "positive"] if c in ds.column_names]

    ds = ds.select_columns(cols_to_keep)

    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))

    load_time = time.time() - t0
    print(f"    + Đã nạp thành công: {len(ds):,} mẫu (mất {load_time:.2f}s, bộ nhớ RAM được tối ưu)")

    return ds


# ==============================================================================
# 3. KHỞI TẠO MÔ HÌNH VÀ HÀM MẤT MÁT (LOSS FUNCTION)
# ==============================================================================
def setup_model_and_loss(
    model_name: str,
    loss_type: str = "mnrl",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    device: str = "cuda",
) -> Tuple[SentenceTransformer, Any]:
    """
    Khởi tạo mô hình SentenceTransformer và hàm Loss phù hợp.

    Args:
        model_name: Tên hoặc đường dẫn mô hình (mặc định paraphrase-multilingual-MiniLM-L12-v2).
        loss_type: 'mnrl' (MultipleNegativesRankingLoss) hoặc 'triplet' (TripletLoss).
        max_seq_length: Giới hạn độ dài token đầu vào (mặc định 256).
        device: Thiết bị chạy ('cuda' hoặc 'cpu').

    Returns:
        (model, loss_function)
    """
    print(f" -> Đang tải mô hình nền: '{model_name}' trên thiết bị '{device}'...")
    model = SentenceTransformer(model_name, device=device)

    # Điều chỉnh max_seq_length phù hợp với độ dài các chunk tài liệu cấu hình điện thoại
    original_seq_len = getattr(model, "max_seq_length", 128)
    model.max_seq_length = max_seq_length
    print(f"    + Cập nhật max_seq_length: {original_seq_len} -> {model.max_seq_length} tokens")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    + Tổng số tham số        : {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"    + Tham số cần train      : {trainable_params:,}")

    # Cấu hình Loss
    if loss_type == "mnrl":
        print("    + Hàm Loss: MultipleNegativesRankingLoss (MNRL)")
        print("      * Cơ chế: Tương phản giữa Anchor với Positive, Hard Negative và toàn bộ in-batch items.")
        loss_fn = losses.MultipleNegativesRankingLoss(model=model)
    elif loss_type == "triplet":
        print("    + Hàm Loss: TripletLoss (Cosine Distance, margin=0.5)")
        loss_fn = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            margin=0.5,
        )
    else:
        raise ValueError(f"Hàm loss không được hỗ trợ: {loss_type}. Chọn 'mnrl' hoặc 'triplet'.")

    return model, loss_fn


# ==============================================================================
# 4. THIẾT LẬP BỘ ĐÁNH GIÁ ĐỊNH KỲ (VALIDATION EVALUATOR)
# ==============================================================================
def create_evaluator(
    val_dataset: Dataset,
    data_type: str = "triplets",
    max_samples: int = 1000,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Optional[Any]:
    """
    Tạo bộ đánh giá định kỳ trên tập kiểm thử (Validation).
    Với Triplets, sử dụng TripletEvaluator để đo lường độ chính xác Cosine Accuracy:
    Tỷ lệ mẫu thỏa mãn: cos(anchor, positive) > cos(anchor, negative).
    """
    if len(val_dataset) == 0:
        return None

    num_eval = min(len(val_dataset), max_samples)
    print(f" -> Khởi tạo Evaluator trên {num_eval:,} mẫu kiểm thử...")

    if data_type == "triplets" and "negative" in val_dataset.column_names:
        eval_anchors = val_dataset["anchor"][:num_eval]
        eval_positives = val_dataset["positive"][:num_eval]
        eval_negatives = val_dataset["negative"][:num_eval]

        evaluator = TripletEvaluator(
            anchors=eval_anchors,
            positives=eval_positives,
            negatives=eval_negatives,
            name="phone_validation",
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return evaluator

    return None


# ==============================================================================
# 5. TIẾN TRÌNH HUẤN LUYỆN CHÍNH (TRAINING PIPELINE)
# ==============================================================================
def run_training(
    model: SentenceTransformer,
    loss_fn: Any,
    train_dataset: Dataset,
    val_dataset: Dataset,
    evaluator: Optional[Any],
    args: argparse.Namespace,
    hw_info: Dict[str, Any],
) -> SentenceTransformer:
    """
    Thực thi tiến trình Fine-tune mô hình Embedding sử dụng SentenceTransformerTrainer.
    """
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = args.fp16 and hw_info["cuda_available"]

    total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    # Cấu hình siêu tham số huấn luyện tối ưu cho RTX 3060
    training_kwargs = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "fp16": use_fp16,
        "eval_steps": args.eval_steps if evaluator is not None else None,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True if evaluator is not None else False,
        "metric_for_best_model": "eval_phone_validation_cosine_accuracy" if evaluator is not None else None,
        "greater_is_better": True if evaluator is not None else None,
        "dataloader_drop_last": True,  # Giữ kích thước mini-batch đồng đều cho MNRL
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": hw_info["cuda_available"],
        "report_to": "none",
        "seed": args.seed,
    }

    eval_strategy_val = "steps" if evaluator is not None else "no"
    try:
        training_args = SentenceTransformerTrainingArguments(
            **training_kwargs,
            eval_strategy=eval_strategy_val,
        )
    except TypeError:
        # Tương thích ngược với transformers cũ mà không gây lỗi Pyrefly
        fallback_kwargs = dict(training_kwargs)
        fallback_kwargs["evaluation_strategy"] = eval_strategy_val
        training_args = SentenceTransformerTrainingArguments(**fallback_kwargs)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if evaluator is not None else None,
        loss=loss_fn,
        evaluator=evaluator,
    )

    print("-" * 75)
    print(" [4/5] BẮT ĐẦU FINE-TUNING EMBEDDING MODEL:")
    print(f" - Số mẫu tập Train        : {len(train_dataset):,}")
    print(f" - Số mẫu tập Val          : {len(val_dataset):,}")
    print(f" - Số Epoch                : {args.epochs}")
    print(f" - Batch Size              : {args.batch_size}")
    print(f" - Learning Rate           : {args.lr}")
    print(f" - Ước tính tổng Steps     : {total_steps:,} steps")
    print(f" - Chế độ FP16             : {'BẬT' if use_fp16 else 'TẮT'}")
    print(f" - Đánh giá mỗi            : {args.eval_steps} steps")
    print(f" - Lưu Checkpoint mỗi      : {args.save_steps} steps (giữ tối đa {args.save_total_limit} bản)")
    print("-" * 75)

    t_start = time.time()
    train_result = trainer.train()
    train_duration = time.time() - t_start

    print("-" * 75)
    print(f" -> Hoàn tất huấn luyện trong {train_duration/60:.2f} phút!")
    if hasattr(train_result, "metrics"):
        print(f" -> Metrics: {train_result.metrics}")

    # Lưu mô hình hoàn thiện cuối cùng
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n -> Đang lưu mô hình xuất xưởng vào: {output_dir.resolve()} ...")
    model.save_pretrained(str(output_dir))
    print(" -> Đã lưu thành công đầy đủ Trọng số (Weights) + Tokenizer + Cấu hình Embedding.")

    return model


# ==============================================================================
# 6. BENCHMARK & DEMO SO SÁNH TRƯỚC VÀ SAU FINE-TUNE (BEFORE VS AFTER)
# ==============================================================================
def run_benchmark_demo(
    fine_tuned_model: SentenceTransformer,
    base_model_name: str,
    device: str = "cpu",
):
    """
    Thực hiện so sánh trực quan giữa Mô hình gốc (Base Model) và Mô hình đã Fine-tune:
    Kiểm tra khả năng phân biệt câu hỏi với Target Document vs Hard Negatives.
    """
    print("=" * 75)
    print(" [5/5] BENCHMARK THỬ NGHIỆM: BASE MODEL VS FINE-TUNED MODEL")
    print("=" * 75)

    print(" -> Đang tải lại mô hình gốc ban đầu để so sánh đối chứng...")
    try:
        base_model = SentenceTransformer(base_model_name, device=device)
    except Exception as e:
        print(f" ! Không thể tải base model để đối chứng ({e}). Bỏ qua benchmark.")
        return

    # Bộ mẫu kiểm thử thực tế về thông số điện thoại
    test_cases = [
        {
            "query": "dung lượng pin và khả năng sạc của Apple iPhone 13 Pro Max",
            "candidates": [
                {
                    "title": "Positive (Pin iPhone 13 Pro Max)",
                    "text": "## Thiết bị: Apple iPhone 13 Pro Max\n- Chuyên mục: Battery\n[Battery]\nNominal Battery Capacity: 4352 mAh\nBattery Type: Li-Ion\nFast Charging: 27W fast charge support\nWireless Charging: MagSafe 15W, Qi 7.5W",
                    "is_positive": True,
                },
                {
                    "title": "Hard Neg 1 (Cùng máy khác mục - Màn hình)",
                    "text": "## Thiết bị: Apple iPhone 13 Pro Max\n- Chuyên mục: Display\n[Display]\nDisplay Diagonal: 6.7 inch\nResolution: 1284x2778\nDisplay Type: Super Retina XDR OLED\nRefresh Rate: 120Hz ProMotion",
                    "is_positive": False,
                },
                {
                    "title": "Hard Neg 2 (Cùng mục Pin nhưng máy Samsung S22 Ultra)",
                    "text": "## Thiết bị: Samsung Galaxy S22 Ultra 5G\n- Chuyên mục: Battery\n[Battery]\nNominal Battery Capacity: 5000 mAh\nFast Charging: 45W fast charge support\nWireless Charging: 15W wireless",
                    "is_positive": False,
                },
            ]
        },
        {
            "query": "Xiaomi 13 dùng chip vi xử lý và card đồ họa gì?",
            "candidates": [
                {
                    "title": "Positive (Chipset Xiaomi 13)",
                    "text": "## Thiết bị: Xiaomi 13\n- Chuyên mục: Processor/Performance\n[Application processor, Chipset]\nCPU Clock: 3.2 GHz\nCPU: Qualcomm Snapdragon 8 Gen 2\nGPU: Qualcomm Adreno 740",
                    "is_positive": True,
                },
                {
                    "title": "Hard Neg 1 (Cùng máy khác mục - Camera)",
                    "text": "## Thiết bị: Xiaomi 13\n- Chuyên mục: Camera\n[Camera Subsystem]\nMain Camera: 50 MP, f/1.8, Leica lens\nOptical Zoom: 3.2x optical zoom\nVideo Recording: 8K at 24fps",
                    "is_positive": False,
                },
                {
                    "title": "Hard Neg 2 (Cùng mục Chip nhưng máy Xiaomi 12)",
                    "text": "## Thiết bị: Xiaomi 12\n- Chuyên mục: Processor/Performance\n[Application processor, Chipset]\nCPU Clock: 3.0 GHz\nCPU: Qualcomm Snapdragon 8 Gen 1\nGPU: Qualcomm Adreno 730",
                    "is_positive": False,
                },
            ]
        }
    ]

    for idx, case in enumerate(test_cases, 1):
        q = case["query"]
        candidates = case["candidates"]
        cand_texts = [c["text"] for c in candidates]

        # Tính toán embedding với Base Model
        q_emb_base = base_model.encode(q, convert_to_tensor=True)
        cand_embs_base = base_model.encode(cand_texts, convert_to_tensor=True)
        scores_base = cos_sim(q_emb_base, cand_embs_base)[0].tolist()

        # Tính toán embedding với Fine-tuned Model
        q_emb_ft = fine_tuned_model.encode(q, convert_to_tensor=True)
        cand_embs_ft = fine_tuned_model.encode(cand_texts, convert_to_tensor=True)
        scores_ft = cos_sim(q_emb_ft, cand_embs_ft)[0].tolist()

        print(f"\n--- TEST CASE #{idx} ---")
        print(f"Câu hỏi (Query): \"{q}\"")
        print("-" * 75)
        print(f"{'Tài liệu ứng viên':<35} | {'Base Model Sim':<16} | {'Fine-tuned Sim':<16} | {'Độ chênh lệch'}")
        print("-" * 75)

        for c, s_base, s_ft in zip(candidates, scores_base, scores_ft):
            delta = s_ft - s_base
            status = " [MỤC TIÊU]" if c["is_positive"] else ""
            label = f"{c['title']}{status}"
            print(f"{label:<35} | {s_base:^16.4f} | {s_ft:^16.4f} | {delta:^+14.4f}")

        # Tính margin chênh lệch giữa Positive và Hard Neg cao nhất
        pos_base = [s for c, s in zip(candidates, scores_base) if c["is_positive"]][0]
        neg_base = max([s for c, s in zip(candidates, scores_base) if not c["is_positive"]])
        margin_base = pos_base - neg_base

        pos_ft = [s for c, s in zip(candidates, scores_ft) if c["is_positive"]][0]
        neg_ft = max([s for c, s in zip(candidates, scores_ft) if not c["is_positive"]])
        margin_ft = pos_ft - neg_ft

        print(f" -> Margin (Positive - Top Negative):")
        print(f"    Base Model: {margin_base:+.4f}  ==>  Fine-tuned: {margin_ft:+.4f} (Độ tách biệt: {margin_ft - margin_base:+.4f})")


# ==============================================================================
# 7. GIAO DIỆN DÒNG LỆNH (CLI) VÀ ĐIỀU PHỐI CHÍNH
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="B4: Fine-tune Embedding Model cho bài toán tra cứu thông số điện thoại"
    )
    # Mô hình và đường dẫn dữ liệu
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Mô hình nền HuggingFace (mặc định: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Thư mục chứa dataset tạo từ B3 (mặc định: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="triplets",
        choices=["triplets", "pairs"],
        help="Loại dữ liệu huấn luyện: 'triplets' (khuyên dùng, có Hard Negatives) hoặc 'pairs'",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mnrl",
        choices=["mnrl", "triplet"],
        help="Hàm mất mát: 'mnrl' (MultipleNegativesRankingLoss) hoặc 'triplet' (TripletLoss)",
    )

    # Siêu tham số huấn luyện (Tối ưu cho RTX 3060)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Kích thước mini-batch (mặc định: {DEFAULT_BATCH_SIZE}, phù hợp RTX 3060 12GB)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Số epoch huấn luyện (mặc định: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Tốc độ học Learning Rate (mặc định: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Tỷ lệ warmup bước đầu (mặc định: 0.1)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Độ dài token tối đa (mặc định: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Bật Mixed Precision FP16 trên GPU (mặc định: True)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Tắt FP16 (chạy full FP32)",
    )

    # Giới hạn mẫu & bước lưu trữ
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số dòng train nạp vào (để chạy thử nghiệm nhanh)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=1000,
        help="Số mẫu kiểm thử tối đa cho Evaluator (mặc định: 1000)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="Tần suất đánh giá validation (steps, mặc định: 250)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Tần suất lưu checkpoint (steps, mặc định: 500)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Tần suất ghi log quá trình huấn luyện (steps, mặc định: 50)",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Số lượng checkpoint tối đa được giữ lại (mặc định: 2)",
    )

    # Thư mục xuất
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Thư mục lưu mô hình hoàn thiện cuối cùng (mặc định: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help=f"Thư mục lưu checkpoint trung gian (mặc định: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (mặc định: 42)",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Chỉ chạy thử nghiệm Benchmark demo (không huấn luyện)",
    )

    args = parser.parse_args()

    # Thiết lập seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 75)
    print(" BƯỚC 4: FINE-TUNE EMBEDDING MODEL CHO THÔNG SỐ ĐIỆN THOẠI")
    print("=" * 75)
    print(f"Mô hình nền (Base)       : {args.model_name}")
    print(f"Thư mục Dataset          : {args.data_dir}")
    print(f"Loại dữ liệu             : {args.data_type.upper()}")
    print(f"Hàm Loss                 : {args.loss.upper()}")
    print(f"Batch size               : {args.batch_size}")
    print(f"Max Sequence Length      : {args.max_seq_length} tokens")
    print(f"Số Epoch                 : {args.epochs}")
    print(f"Thư mục lưu mô hình      : {args.output_dir}")
    if args.limit:
        print(f"Giới hạn mẫu thử nghiệm  : {args.limit:,} mẫu")

    # 1. Kiểm tra phần cứng
    hw = check_hardware_and_environment()
    print_hardware_summary(hw, args.batch_size, args.fp16)

    # Chế độ Demo-only (kiểm tra mô hình đã lưu sẵn)
    if args.demo_only:
        out_p = Path(args.output_dir)
        if out_p.exists():
            print(f" -> Tải mô hình đã lưu từ {out_p.resolve()} để chạy demo...")
            ft_model = SentenceTransformer(str(out_p), device=hw["device"])
        else:
            print(f" -> Chưa tìm thấy mô hình tại {out_p}. Dùng base model để kiểm tra demo...")
            ft_model = SentenceTransformer(args.model_name, device=hw["device"])
        run_benchmark_demo(ft_model, args.model_name, device=hw["device"])
        return

    # 2. Xác định các file dữ liệu
    data_dir = Path(args.data_dir)
    if args.data_type == "triplets":
        train_file = data_dir / "train_triplets.jsonl"
        val_file = data_dir / "val_triplets.jsonl"
    else:
        train_file = data_dir / "train_pairs.jsonl"
        val_file = data_dir / "val_pairs.jsonl"

    print("\n[2/5] NẠP DỮ LIỆU:")
    train_dataset = load_dataset_from_jsonl(train_file, data_type=args.data_type, limit=args.limit)
    val_dataset = load_dataset_from_jsonl(val_file, data_type=args.data_type, limit=args.eval_samples)

    # 3. Khởi tạo mô hình và loss
    print("\n[3/5] KHỞI TẠO MÔ HÌNH VÀ LOSS FUNCTION:")
    model, loss_fn = setup_model_and_loss(
        model_name=args.model_name,
        loss_type=args.loss,
        max_seq_length=args.max_seq_length,
        device=hw["device"],
    )

    # 4. Khởi tạo Evaluator
    evaluator = create_evaluator(
        val_dataset=val_dataset,
        data_type=args.data_type,
        max_samples=args.eval_samples,
        batch_size=args.batch_size,
    )

    # 5. Tiến hành huấn luyện
    fine_tuned_model = run_training(
        model=model,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        evaluator=evaluator,
        args=args,
        hw_info=hw,
    )

    # 6. Benchmark so sánh Before vs After
    run_benchmark_demo(fine_tuned_model, args.model_name, device=hw["device"])

    print("\n" + "=" * 75)
    print(" HOÀN TẤT THÀNH CÔNG BƯỚC B4!")
    print(f" Mô hình Fine-tune đã sẵn sàng tại: {Path(args.output_dir).resolve()}")
    print(" Bạn có thể nạp mô hình này trong Bước B5 (Xây dựng Vector DB & RAG Search):")
    print(f"   from sentence_transformers import SentenceTransformer")
    print(f"   model = SentenceTransformer(r\"{Path(args.output_dir).resolve()}\")")
    print("=" * 75)


if __name__ == "__main__":
    main()
