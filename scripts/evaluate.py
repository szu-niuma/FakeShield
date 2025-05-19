from collections import defaultdict
import cv2
import json
import numpy as np
import os
import concurrent.futures
from functools import partial
from pathlib import Path
from tqdm import tqdm


def compute_global_metrics(pred_masks_list, gt_masks_list):
    # 初始化存储各样本指标的列表
    f1_list = []
    precision_list = []
    recall_list = []
    iou_list = []
    acc_list = []

    eps = np.finfo(np.float32).eps  # 防止除以零的小量

    # 逐张处理每张图片
    for pred_mask, gt_mask in zip(pred_masks_list, gt_masks_list):
        # 转换为numpy数组（若输入为列表或其他类型）
        pred_mask = np.array(pred_mask)
        gt_mask = np.array(gt_mask)

        # 数据归一化（逐张处理）
        if pred_mask.max() > 1:
            pred_mask = pred_mask / 255.0
        if gt_mask.max() > 1:
            gt_mask = gt_mask / 255.0

        # 二值化处理
        pred_binary = (pred_mask > 0.5).astype(np.bool_)
        gt_binary = (gt_mask > 0.5).astype(np.bool_)

        # 计算当前样本的统计量
        tp = np.sum(np.logical_and(pred_binary, gt_binary))
        fp = np.sum(np.logical_and(pred_binary, ~gt_binary))
        fn = np.sum(np.logical_and(~pred_binary, gt_binary))
        tn = np.sum(np.logical_and(~pred_binary, ~gt_binary))

        # 计算指标（处理分母为零的特殊情况）
        if (tp + fp) == 0:
            precision = 1.0
        else:
            precision = tp / (tp + fp + eps)

        if (tp + fn) == 0:
            recall = 1.0
        else:
            recall = tp / (tp + fn + eps)

        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall + eps)

        iou = tp / (tp + fp + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)

        # 记录当前样本的指标
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        iou_list.append(iou)
        acc_list.append(acc)

    # 计算所有样本的平均指标
    avg_f1 = np.mean(f1_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_iou = np.mean(iou_list)
    avg_acc = np.mean(acc_list)

    return {"seg_f1": avg_f1, "seg_precision": avg_precision, "seg_recall": avg_recall, "seg_iou": avg_iou, "seg_acc": avg_acc}


def compute_metrics(pred_masks_list, gt_masks_list):
    # 统一输入为列表格式
    pred_masks = np.array(pred_masks_list, dtype=object)
    gt_masks = np.array(gt_masks_list, dtype=object)

    # 输入验证和预处理
    assert pred_masks.shape == gt_masks.shape, "预测和真实标签形状不一致"
    eps = np.finfo(np.float32).eps

    # 数据归一化 (0~1)
    pred = pred_masks / 255.0 if pred_masks.max() > 1 else pred_masks.astype(np.float32)
    gt = gt_masks / 255.0 if gt_masks.max() > 1 else gt_masks.astype(np.float32)

    # 二值化处理
    pred_binary = pred > 0.5
    gt_binary = gt > 0.5

    # 维度处理（保留批次维度）
    axis = tuple(range(1, pred_binary.ndim))

    # 核心指标计算
    intersection = np.sum(pred_binary & gt_binary, axis=axis)
    union = np.sum(pred_binary | gt_binary, axis=axis)

    # IoU计算（处理除零）
    iou = np.divide(intersection, union + eps, where=(union > 0), out=np.ones_like(intersection, dtype=np.float32))

    # Precision计算
    pred_sums = np.sum(pred_binary, axis=axis)
    gt_sums = np.sum(gt_binary, axis=axis)
    precision = np.divide(
        intersection, pred_sums + eps, where=(pred_sums > 0) | (gt_sums > 0), out=np.where((pred_sums == 0) & (gt_sums == 0), 1.0, 0.0)
    )

    # Recall计算（真实无正样本时设为NaN）
    recall = np.divide(intersection, gt_sums + eps, where=(gt_sums > 0), out=np.where(gt_sums == 0, np.nan, 1.0))

    # F1计算（忽略NaN和零分母）
    denominator = precision + recall
    f1 = np.zeros_like(denominator)
    valid_mask = (denominator > 0) & ~np.isnan(denominator)
    f1[valid_mask] = 2 * precision[valid_mask] * recall[valid_mask] / denominator[valid_mask]

    # 准确率计算（全局像素级）
    correct = (pred_binary == gt_binary).astype(np.float32)
    total_correct = np.sum(correct)
    total_pixels = np.prod(correct.shape)
    accuracy = total_correct / total_pixels

    # 过滤无效Recall值
    valid_recall = recall[~np.isnan(recall)]

    return {
        "seg_f1": np.mean(f1),
        "seg_iou": np.mean(iou),
        "seg_precision": np.mean(precision),
        "seg_recall": np.mean(valid_recall) if valid_recall.size > 0 else np.nan,
        "seg_acc": accuracy,  # 修正为全局正确率
    }


def read_json(json_path):
    """添加文件存在性检查"""
    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def process_item(values, real_pred_dir, fake_pred_dir):
    """添加异常处理及路径验证"""
    try:
        # ===== 处理真实图像 =====
        real_img_path = Path(values["real_img"])
        if not real_img_path.exists():
            raise FileNotFoundError(f"真实图像不存在: {real_img_path}")

        # 读取预测掩码
        real_pred_mask_path = real_pred_dir / real_img_path.name
        if not real_pred_mask_path.exists():
            raise FileNotFoundError(f"真实预测掩码不存在: {real_pred_mask_path}")
        real_pred_mask = cv2.imread(str(real_pred_mask_path), cv2.IMREAD_GRAYSCALE)

        # 生成全零真值（根据实际需求修改）
        real_gt_mask = np.zeros_like(real_pred_mask)

        # ===== 处理伪造图像 =====
        fake_img_path = Path(values["fake_img"])
        fake_mask_path = Path(values["fake_mask"])
        if not fake_img_path.exists():
            raise FileNotFoundError(f"伪造图像不存在: {fake_img_path}")
        if not fake_mask_path.exists():
            raise FileNotFoundError(f"伪造真值掩码不存在: {fake_mask_path}")

        # 读取预测掩码
        fake_pred_mask_path = fake_pred_dir / fake_img_path.name
        if not fake_pred_mask_path.exists():
            raise FileNotFoundError(f"伪造预测掩码不存在: {fake_pred_mask_path}")
        fake_pred_mask = cv2.imread(str(fake_pred_mask_path), cv2.IMREAD_GRAYSCALE)

        # 读取真值掩码
        fake_gt_mask = cv2.imread(str(fake_mask_path), cv2.IMREAD_GRAYSCALE)

        # real_res = compute_global_metrics(real_pred_mask, real_gt_mask)
        # fake_res = compute_global_metrics(fake_pred_mask, fake_gt_mask)

        # ===== 合并结果 =====
        # return {key: (real_res[key] + fake_res[key]) / 2 for key in real_res}
        return real_pred_mask, real_gt_mask, fake_pred_mask, fake_gt_mask
    except Exception as e:
        print(f"处理 {values.get('fake_img', '未知条目')} 时出错: {str(e)}")
        raise


def main(target_json, real_pred_dir, fake_pred_dir, max_workers=None):
    """优化并行任务管理"""
    data = read_json(target_json)

    # 自动计算最佳线程数
    if max_workers is None:
        max_workers = min(len(data), (os.cpu_count() or 1) * 2)

    real_pred_mask, real_gt_mask, fake_pred_mask, fake_gt_mask = [], [], [], []
    # 使用进程池处理CPU密集型任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item, Path(real_pred_dir), Path(fake_pred_dir)) for item in data.values()]

        # 进度显示优化
        with tqdm(total=len(futures), desc="处理进度", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    real_pred_mask.append(res[0])
                    real_gt_mask.append(res[1])
                    fake_pred_mask.append(res[2])
                    fake_gt_mask.append(res[3])
                except Exception as exc:
                    print(f"\n处理错误: {exc}")
                else:
                    pbar.update(1)

    # 计算最终指标
    real_res = compute_global_metrics(real_pred_mask, real_gt_mask)
    fake_res = compute_global_metrics(fake_pred_mask, fake_gt_mask)
    # 打印格式化结果
    print("\nReal最终评估指标:")
    for metric, value in real_res.items():
        print(f"{metric:15}: {value:.4f}")
    print("\nFake最终评估指标:")
    for metric, value in fake_res.items():
        print(f"{metric:15}: {value:.4f}")
    print("\n总是指标")
    for metric, value in real_res.items():
        print(f"{metric:15}: {(value + fake_res[metric]) / 2:.4f}")


if __name__ == "__main__":
    # 使用更清晰的路径管理
    BASE_DIR = Path("/home/yuyangxin/data")

    main(
        target_json=BASE_DIR / "dataset/MagicBrush/record.json",
        real_pred_dir=BASE_DIR / "FakeShield/playground/MFLM_output/real_img",
        fake_pred_dir=BASE_DIR / "FakeShield/playground/MFLM_output/fake_img",
        max_workers=32,
    )
