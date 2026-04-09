"""
Baseline pipeline for equipment utilization tracking.

What this script does:
1) Runs object detection + tracking on a video.
2) Estimates motion inside each track box and infers ACTIVE/INACTIVE.
3) Assigns a baseline activity label (DIGGING, SWINGING_LOADING, DUMPING, WAITING).
4) Accumulates tracked/active/idle/dwell times per equipment ID.
5) Exports:
   - Annotated video
   - Frame-level JSONL events
   - Per-equipment summary CSV

Notes:
- This baseline includes lightweight reassociation (IoU + appearance) to reduce ID switches.
- The next milestone will upgrade to stronger global ReID for long occlusions and camera changes.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


@dataclass
class EquipmentTimer:
    equipment_id: str
    equipment_class: str
    total_tracked_seconds: float = 0.0
    total_active_seconds: float = 0.0
    total_idle_seconds: float = 0.0
    last_timestamp_s: Optional[float] = None
    last_state: Optional[str] = None
    inactive_start_s: Optional[float] = None
    current_activity: str = "WAITING"
    motion_source: str = "none"

    def update(self, timestamp_s: float, state: str, activity: str, motion_source: str) -> None:
        if self.last_timestamp_s is not None:
            dt = max(0.0, timestamp_s - self.last_timestamp_s)
            self.total_tracked_seconds += dt
            if self.last_state == "ACTIVE":
                self.total_active_seconds += dt
            else:
                self.total_idle_seconds += dt

        if state == "INACTIVE" and self.last_state != "INACTIVE":
            self.inactive_start_s = timestamp_s
        if state == "ACTIVE":
            self.inactive_start_s = None

        self.last_timestamp_s = timestamp_s
        self.last_state = state
        self.current_activity = activity
        self.motion_source = motion_source

    @property
    def utilization_percent(self) -> float:
        if self.total_tracked_seconds <= 0:
            return 0.0
        return 100.0 * (self.total_active_seconds / self.total_tracked_seconds)

    def current_dwell_seconds(self, now_s: float) -> float:
        if self.last_state != "INACTIVE" or self.inactive_start_s is None:
            return 0.0
        return max(0.0, now_s - self.inactive_start_s)


def seconds_to_timestamp_string(seconds: float) -> str:
    total = int(seconds)
    ms = int(round((seconds - total) * 1000))
    if ms == 1000:
        total += 1
        ms = 0
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def normalize_label(label: str) -> str:
    return label.lower().replace("_", " ").strip()


def class_name_allowed(cls_name: str, allowed_names: Optional[Sequence[str]]) -> bool:
    if not allowed_names:
        return True
    norm_cls = normalize_label(cls_name)
    for name in allowed_names:
        norm_name = normalize_label(name)
        if not norm_name:
            continue
        if norm_cls == norm_name or norm_name in norm_cls or norm_cls in norm_name:
            return True
    return False


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float).tolist()
    bx1, by1, bx2, by2 = box_b.astype(float).tolist()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 1e-9:
        return 0.0
    return float(inter_area / union)


def box_center(box_xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def box_area(box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    return max(1.0, x2 - x1) * max(1.0, y2 - y1)


def box_diag(box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    return float(np.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1)))


def box_aspect_ratio(box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return float(w / h)


def box_containment_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float).tolist()
    bx1, by1, bx2, by2 = box_b.astype(float).tolist()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    min_area = min(box_area(box_a), box_area(box_b))
    if min_area <= 1e-9:
        return 0.0
    return float(inter_area / min_area)


def x_overlap_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, _, ax2, _ = box_a.astype(float).tolist()
    bx1, _, bx2, _ = box_b.astype(float).tolist()
    inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    min_w = min(max(1.0, ax2 - ax1), max(1.0, bx2 - bx1))
    return float(inter_x / min_w)


def y_overlap_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    _, ay1, _, ay2 = box_a.astype(float).tolist()
    _, by1, _, by2 = box_b.astype(float).tolist()
    inter_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    min_h = min(max(1.0, ay2 - ay1), max(1.0, by2 - by1))
    return float(inter_y / min_h)


def center_distance_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    cax, cay = box_center(box_a)
    cbx, cby = box_center(box_b)
    dist = np.hypot(cax - cbx, cay - cby)
    ref = max(1.0, max(box_diag(box_a), box_diag(box_b)))
    return float(dist / ref)


def center_x_distance_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    cax, _ = box_center(box_a)
    cbx, _ = box_center(box_b)
    ax1, _, ax2, _ = box_a.astype(float).tolist()
    bx1, _, bx2, _ = box_b.astype(float).tolist()
    ref_w = max(1.0, max(ax2 - ax1, bx2 - bx1))
    return float(abs(cax - cbx) / ref_w)


def vertical_gap_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float).tolist()
    bx1, by1, bx2, by2 = box_b.astype(float).tolist()

    a_h = max(1.0, ay2 - ay1)
    b_h = max(1.0, by2 - by1)

    # Order boxes by vertical position and measure normalized vertical gap.
    if ay1 <= by1:
        upper_y2 = ay2
        lower_y1 = by1
    else:
        upper_y2 = by2
        lower_y1 = ay1

    gap = max(0.0, lower_y1 - upper_y2)
    return float(gap / max(1.0, min(a_h, b_h)))


def is_vertical_split_pair(box_a: np.ndarray, box_b: np.ndarray, args: argparse.Namespace) -> bool:
    x_ov = x_overlap_ratio(box_a, box_b)
    cx_dist = center_x_distance_ratio(box_a, box_b)
    v_gap = vertical_gap_ratio(box_a, box_b)
    area_ratio = max(box_area(box_a), box_area(box_b)) / max(1.0, min(box_area(box_a), box_area(box_b)))

    return (
        x_ov >= args.dup_vertical_x_overlap_thr
        and cx_dist <= args.dup_vertical_center_x_ratio
        and v_gap <= args.dup_vertical_gap_ratio
        and area_ratio <= args.dup_vertical_max_area_ratio
    )


def detections_are_duplicates(det_a: Dict[str, object], det_b: Dict[str, object], args: argparse.Namespace) -> bool:
    box_a = det_a["box_xyxy"]
    box_b = det_b["box_xyxy"]

    iou = bbox_iou(box_a, box_b)
    ctr_dist = center_distance_ratio(box_a, box_b)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    area_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))

    # Keep IoU-only suppression conservative to avoid collapsing adjacent machines during occlusion.
    if (
        iou >= args.dup_iou_thr
        and ctr_dist <= (args.dup_center_dist_ratio * 0.75)
        and area_ratio <= min(args.dup_max_area_ratio, 2.2)
    ):
        return True

    contain = box_containment_ratio(box_a, box_b)
    if contain >= args.dup_containment_thr:
        return True

    if is_vertical_split_pair(box_a, box_b, args):
        return True

    x_ov = x_overlap_ratio(box_a, box_b)
    y_ov = y_overlap_ratio(box_a, box_b)

    if (
        x_ov >= args.dup_x_overlap_thr
        and y_ov >= args.dup_y_overlap_thr
        and ctr_dist <= args.dup_center_dist_ratio
        and area_ratio <= args.dup_max_area_ratio
    ):
        return True

    return False


def suppress_duplicate_detections(detections: Sequence[Dict[str, object]], args: argparse.Namespace) -> list[Dict[str, object]]:
    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda d: float(d["box_area"]) * (1.0 + 0.15 * float(d["score"])),
        reverse=True,
    )

    kept: list[Dict[str, object]] = []
    for det in ordered:
        cls_name = str(det["cls_name"])
        duplicate = False
        for prev in kept:
            if str(prev["cls_name"]) != cls_name:
                continue
            if detections_are_duplicates(det, prev, args):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    return kept


def likely_same_equipment_in_frame(box_a: np.ndarray, box_b: np.ndarray, args: argparse.Namespace) -> bool:
    """
    Conservative same-frame duplicate check used after tracker output.
    This specifically targets split boxes on one machine (e.g., upper/lower body)
    and avoids creating a new equipment ID for the second fragment.
    """
    iou = bbox_iou(box_a, box_b)
    contain = box_containment_ratio(box_a, box_b)
    x_ov = x_overlap_ratio(box_a, box_b)
    y_ov = y_overlap_ratio(box_a, box_b)
    ctr_dist = center_distance_ratio(box_a, box_b)
    area_ratio = max(box_area(box_a), box_area(box_b)) / max(1.0, min(box_area(box_a), box_area(box_b)))

    # Be slightly more permissive than detector-level suppression to avoid
    # spawning a second equipment ID for upper/lower body fragments.
    iou_gate = max(0.32, args.dup_iou_thr * 0.90)
    contain_gate = max(0.72, args.dup_containment_thr * 0.90)
    x_gate = max(0.62, args.dup_x_overlap_thr * 0.90)
    y_gate = max(0.28, args.dup_y_overlap_thr * 0.85)
    ctr_gate = min(0.62, args.dup_center_dist_ratio * 1.20)
    area_gate = min(5.5, args.dup_max_area_ratio * 1.60)

    if iou >= iou_gate:
        return True
    if contain >= contain_gate:
        return True
    if is_vertical_split_pair(box_a, box_b, args):
        return True
    if x_ov >= x_gate and y_ov >= y_gate and ctr_dist <= ctr_gate and area_ratio <= area_gate:
        return True
    # Additional fallback for near-collinear overlaps with moderate center distance.
    if x_ov >= 0.55 and ctr_dist <= 0.35 and area_ratio <= 6.0:
        return True
    return False


def cosine_similarity(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def extract_appearance_feature(frame: np.ndarray, box_xyxy: np.ndarray, bins: int) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int).tolist()
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)

    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    if hist is None:
        return None
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    return hist


def estimate_global_shift(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    scale: float,
) -> Tuple[float, float]:
    if prev_gray is None:
        return 0.0, 0.0
    if scale <= 0:
        return 0.0, 0.0

    try:
        if scale < 0.999:
            h, w = curr_gray.shape[:2]
            ws = max(32, int(round(w * scale)))
            hs = max(32, int(round(h * scale)))
            prev_small = cv2.resize(prev_gray, (ws, hs), interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(curr_gray, (ws, hs), interpolation=cv2.INTER_AREA)
            (dx_s, dy_s), _ = cv2.phaseCorrelate(prev_small.astype(np.float32), curr_small.astype(np.float32))
            inv = 1.0 / scale
            return float(dx_s * inv), float(dy_s * inv)

        (dx, dy), _ = cv2.phaseCorrelate(prev_gray.astype(np.float32), curr_gray.astype(np.float32))
        return float(dx), float(dy)
    except cv2.error:
        return 0.0, 0.0


def dominant_flow_angle_deg(flow_xy: np.ndarray, mag: np.ndarray) -> float:
    """Return dominant flow angle in image coordinates (0..360, y axis points down)."""
    if flow_xy.size == 0 or mag.size == 0:
        return float("nan")

    p75 = float(np.percentile(mag, 75))
    dyn = float(np.mean(mag) + 0.20)
    thr = max(p75, dyn)
    mask = mag >= thr
    if int(mask.sum()) < 8:
        return float("nan")

    vx = float(np.mean(flow_xy[..., 0][mask]))
    vy = float(np.mean(flow_xy[..., 1][mask]))
    if float(np.hypot(vx, vy)) < 1e-3:
        return float("nan")

    return float((np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0)


def angle_in_ranges(angle_deg: float, ranges_deg: Sequence[Tuple[float, float]]) -> bool:
    if angle_deg is None or np.isnan(angle_deg):
        return False
    a = float(angle_deg % 360.0)
    for low, high in ranges_deg:
        lo = float(low % 360.0)
        hi = float(high % 360.0)
        if lo <= hi:
            if lo <= a <= hi:
                return True
        else:
            if a >= lo or a <= hi:
                return True
    return False


def compute_motion_scores(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    box_xyxy: np.ndarray,
    global_shift_xy: Tuple[float, float] = (0.0, 0.0),
    border_margin: float = 0.10,
) -> Dict[str, float]:
    """
    Returns magnitude, occupancy and directional-flow features for class-specific rules.
    Image-coordinate angles are used: 90°=down, 270°=up.
    """
    empty = {
        "full_motion": 0.0,
        "arm_motion": 0.0,
        "base_motion": 0.0,
        "full_occ": 0.0,
        "arm_occ": 0.0,
        "base_occ": 0.0,
        "full_angle_deg": float("nan"),
        "arm_angle_deg": float("nan"),
        "base_angle_deg": float("nan"),
        "top40_motion": 0.0,
        "top40_occ": 0.0,
        "top40_angle_deg": float("nan"),
        "top60_motion": 0.0,
        "top60_occ": 0.0,
        "top60_angle_deg": float("nan"),
        "bottom50_motion": 0.0,
        "bottom50_occ": 0.0,
        "bottom50_angle_deg": float("nan"),
    }
    if prev_gray is None:
        return empty

    h, w = curr_gray.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int).tolist()
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)

    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return empty

    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]

    flow = cv2.calcOpticalFlowFarneback(
        prev_roi,
        curr_roi,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=15,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )

    dx, dy = global_shift_xy
    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        flow[..., 0] -= float(dx)
        flow[..., 1] -= float(dy)

    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Use inner ROI statistics to reduce boundary contamination from neighboring machines.
    margin = float(max(0.0, min(0.45, border_margin)))
    mh = int(round(mag.shape[0] * margin))
    mw = int(round(mag.shape[1] * margin))
    y1c = mh
    y2c = max(y1c + 1, mag.shape[0] - mh)
    x1c = mw
    x2c = max(x1c + 1, mag.shape[1] - mw)
    core = mag[y1c:y2c, x1c:x2c]
    core_flow = flow[y1c:y2c, x1c:x2c, :]
    if core.size < 16:
        core = mag
        core_flow = flow

    def region_stats(region_mag: np.ndarray, region_flow: np.ndarray) -> Tuple[float, float, float]:
        if region_mag.size == 0:
            return 0.0, 0.0, float("nan")
        med = float(np.median(region_mag))
        q75 = float(np.percentile(region_mag, 75))
        q90 = float(np.percentile(region_mag, 90))
        score = max(med, 0.72 * q75, 0.58 * q90)
        dynamic_thr = med + 0.35
        occ = float(np.mean(region_mag >= dynamic_thr))
        ang = dominant_flow_angle_deg(region_flow, region_mag)
        return score, occ, ang

    full_motion, full_occ, full_ang = region_stats(core, core_flow)

    core_h = max(1, core.shape[0])
    top35_end = max(1, int(round(0.35 * core_h)))
    top40_end = max(1, int(round(0.40 * core_h)))
    top60_end = max(1, int(round(0.60 * core_h)))
    bottom50_start = min(core_h - 1, int(round(0.50 * core_h)))

    arm_mag = core[:top35_end, :]
    arm_flow = core_flow[:top35_end, :, :]
    base_mag = core[bottom50_start:, :]
    base_flow = core_flow[bottom50_start:, :, :]

    top40_mag = core[:top40_end, :]
    top40_flow = core_flow[:top40_end, :, :]
    top60_mag = core[:top60_end, :]
    top60_flow = core_flow[:top60_end, :, :]
    bottom50_mag = core[bottom50_start:, :]
    bottom50_flow = core_flow[bottom50_start:, :, :]

    arm_motion, arm_occ, arm_ang = region_stats(arm_mag, arm_flow)
    base_motion, base_occ, base_ang = region_stats(base_mag, base_flow)
    top40_motion, top40_occ, top40_ang = region_stats(top40_mag, top40_flow)
    top60_motion, top60_occ, top60_ang = region_stats(top60_mag, top60_flow)
    bottom50_motion, bottom50_occ, bottom50_ang = region_stats(bottom50_mag, bottom50_flow)

    return {
        "full_motion": full_motion,
        "arm_motion": arm_motion,
        "base_motion": base_motion,
        "full_occ": full_occ,
        "arm_occ": arm_occ,
        "base_occ": base_occ,
        "full_angle_deg": full_ang,
        "arm_angle_deg": arm_ang,
        "base_angle_deg": base_ang,
        "top40_motion": top40_motion,
        "top40_occ": top40_occ,
        "top40_angle_deg": top40_ang,
        "top60_motion": top60_motion,
        "top60_occ": top60_occ,
        "top60_angle_deg": top60_ang,
        "bottom50_motion": bottom50_motion,
        "bottom50_occ": bottom50_occ,
        "bottom50_angle_deg": bottom50_ang,
    }


def infer_state_and_source(
    full_motion: float,
    arm_motion: float,
    base_motion: float,
    center_motion: float,
    full_occ: float,
    arm_occ: float,
    base_occ: float,
    full_active_thr: float,
    arm_active_thr: float,
    base_active_thr: float,
    center_active_thr: float,
    micro_motion_ratio: float,
    min_motion_occupancy: float,
    min_part_motion_occupancy: float,
) -> Tuple[str, str]:
    occ_full = max(0.0, min(1.0, min_motion_occupancy))
    occ_part = max(0.0, min(1.0, min_part_motion_occupancy))

    full_active = (full_motion >= full_active_thr) and (full_occ >= occ_full)
    center_active = center_motion >= center_active_thr

    # FIX2: Require minimal whole-body floor, but keep it low enough for
    # stationary-body arm-only work (only top ~45% of box ever moves).
    part_full_floor = max(0.03, full_active_thr * 0.14)
    arm_candidate = (arm_motion >= arm_active_thr) and (arm_occ >= occ_part)
    base_candidate = (base_motion >= base_active_thr) and (base_occ >= occ_part)

    arm_dominant = (
        arm_motion >= (base_motion * 1.05)
        or (arm_motion - base_motion) >= max(0.05, arm_active_thr * 0.08)
    )
    base_dominant = (
        base_motion >= (arm_motion * 1.05)
        or (base_motion - arm_motion) >= max(0.05, base_active_thr * 0.08)
    )

    # Keep articulated-motion sensitivity: strong localized arm/base movement can be ACTIVE
    # even when whole-box motion is modest.
    # FIX3: arm_dominant alone is sufficient evidence — stationary body + moving arm
    # produces low full_motion, so requiring the old floor blocked arm-only ACTIVE.
    arm_active = arm_candidate and (
        (full_motion >= part_full_floor)
        or center_active
        or arm_dominant
    )
    base_active = base_candidate and (
        (full_motion >= part_full_floor)
        or center_active
        or base_dominant
    )

    # Promote subtle partial movement to ACTIVE (requested behavior):
    # any small movement in a machine part should count as ACTIVE.
    micro_ratio = max(0.05, min(1.0, micro_motion_ratio))
    full_micro = (full_motion >= (full_active_thr * micro_ratio)) and (full_occ >= max(0.025, occ_full * 0.85))
    arm_micro = (arm_motion >= (arm_active_thr * micro_ratio)) and (arm_occ >= max(0.02, occ_part * 0.8))
    base_micro = (base_motion >= (base_active_thr * micro_ratio)) and (base_occ >= max(0.02, occ_part * 0.8))

    # Reject uniform low-level jitter by requiring localized micro contrast
    # when promoting subtle partial movement to ACTIVE.
    contrast_ratio = 1.08
    micro_delta = max(0.04, 0.06 * max(1.0, full_active_thr))
    localized_arm = arm_micro and (
        arm_motion >= (base_motion * contrast_ratio)
        or (arm_motion - base_motion) >= micro_delta
    ) and (arm_occ >= max(0.03, occ_part * 0.9))
    localized_base = base_micro and (
        base_motion >= (arm_motion * contrast_ratio)
        or (base_motion - arm_motion) >= micro_delta
    ) and (base_occ >= max(0.03, occ_part * 0.9))

    # Extra-sensitive pulse gates for subtle articulated movement.
    # These are intentionally more permissive than localized_* to avoid
    # missing true arm-only movement with tiny occupied area.
    # FIX4: separate floors — arm-only on stationary machine has low full_motion;
    # a unified 0.20 floor was blocking all arm-only ACTIVE pulses.
    pulse_full_floor_arm  = max(0.03, full_active_thr * 0.09)   # ~0.06 at default
    pulse_full_floor_base = max(0.07, full_active_thr * 0.22)   # ~0.15 at default
    arm_pulse = (
        arm_motion >= (arm_active_thr * micro_ratio * 0.82)
        and arm_occ >= max(0.018, occ_part * 0.65)
        and full_motion >= pulse_full_floor_arm
        and (
            arm_motion >= (base_motion * 1.03)
            or (arm_motion - base_motion) >= (micro_delta * 0.55)
        )
    )
    base_pulse = (
        base_motion >= (base_active_thr * micro_ratio * 0.88)
        and base_occ >= max(0.02, occ_part * 0.70)
        and full_motion >= pulse_full_floor_base
        and (
            base_motion >= (arm_motion * 1.03)
            or (base_motion - arm_motion) >= (micro_delta * 0.55)
        )
    )
    structured_full = (
        full_micro
        and full_occ >= max(0.03, occ_full * 0.70)
        and (max(arm_motion, base_motion) >= (min(arm_active_thr, base_active_thr) * micro_ratio * 0.9))
    )

    if not (full_active or arm_active or base_active or center_active):
        if localized_arm or arm_pulse:
            arm_active = True
        elif localized_base or base_pulse or structured_full:
            base_active = True

    if full_active:
        base_active = True
    elif center_active and full_occ >= max(0.03, occ_full * 0.75) and (
        localized_arm
        or localized_base
        or arm_pulse
        or base_pulse
        or structured_full
        or full_micro
    ):
        # Center jitter alone should not flip to ACTIVE.
        base_active = True

    if arm_active and base_active:
        return "ACTIVE", "both"
    if arm_active and not base_active:
        return "ACTIVE", "arm_only"
    if base_active and not arm_active:
        return "ACTIVE", "base_only"
    return "INACTIVE", "none"


def infer_activity(
    state: str,
    motion_source: str,
    equipment_class: str,
    flow_features: Dict[str, float],
    full_active_thr: float,
    arm_active_thr: float,
    base_active_thr: float,
    center_motion: float,
    center_active_thr: float,
) -> str:
    if state == "INACTIVE":
        return "WAITING"

    cls = normalize_label(equipment_class)

    down_range = [(60, 120)]
    horizontal_ranges = [(0, 30), (150, 210), (330, 360)]
    up_range = [(240, 300)]

    full_motion = float(flow_features.get("full_motion", 0.0))
    arm_motion = float(flow_features.get("arm_motion", 0.0))
    base_motion = float(flow_features.get("base_motion", 0.0))
    full_occ = float(flow_features.get("full_occ", 0.0))
    arm_occ = float(flow_features.get("arm_occ", 0.0))
    base_occ = float(flow_features.get("base_occ", 0.0))

    full_ang = float(flow_features.get("full_angle_deg", float("nan")))
    arm_ang = float(flow_features.get("arm_angle_deg", float("nan")))
    top40_ang = float(flow_features.get("top40_angle_deg", float("nan")))
    top60_ang = float(flow_features.get("top60_angle_deg", float("nan")))
    bed_ang = float(flow_features.get("bottom50_angle_deg", float("nan")))

    top40_motion = float(flow_features.get("top40_motion", 0.0))
    top40_occ = float(flow_features.get("top40_occ", 0.0))
    top60_motion = float(flow_features.get("top60_motion", 0.0))
    top60_occ = float(flow_features.get("top60_occ", 0.0))
    bed_motion = float(flow_features.get("bottom50_motion", 0.0))
    bed_occ = float(flow_features.get("bottom50_occ", 0.0))

    articulated_classes = {"excavator", "backhoe loader"}
    wheel_loader_classes = {"wheel loader"}
    dump_truck_classes = {"dump truck"}
    dozer_classes = {"dozer"}
    crane_classes = {"mobile crane", "tower crane"}
    body_only_working_classes = {"grader", "compactor", "cement truck"}

    if cls in articulated_classes:
        arm_signal = arm_motion >= (arm_active_thr * 0.52) and arm_occ >= 0.02
        if arm_signal and angle_in_ranges(arm_ang, down_range):
            return "DIGGING"
        if arm_signal and angle_in_ranges(arm_ang, up_range):
            return "DUMPING"
        if arm_signal and angle_in_ranges(arm_ang, horizontal_ranges):
            return "SWINGING_LOADING"
        if motion_source == "arm_only":
            return "DIGGING"
        if motion_source in {"both", "base_only"}:
            return "SWINGING_LOADING"
        return "SWINGING_LOADING"

    if cls in wheel_loader_classes:
        body_signal = (
            (full_motion >= full_active_thr * 0.90 and full_occ >= 0.03)
            or center_motion >= center_active_thr
        )
        bucket_signal = top40_motion >= (arm_active_thr * 0.56) and top40_occ >= 0.02
        if bucket_signal and angle_in_ranges(top40_ang, up_range):
            return "DUMPING"
        if body_signal and not bucket_signal:
            return "MOVING"
        if bucket_signal:
            return "SWINGING_LOADING"
        return "SWINGING_LOADING"

    if cls in dump_truck_classes:
        body_signal = (
            (full_motion >= full_active_thr * 0.95 and full_occ >= 0.03)
            or center_motion >= center_active_thr
        )
        bed_signal = bed_motion >= (base_active_thr * 0.55) and bed_occ >= 0.02
        if bed_signal and angle_in_ranges(bed_ang, up_range):
            return "DUMPING"
        if body_signal:
            return "MOVING"
        return "WAITING"

    if cls in dozer_classes:
        body_signal = (
            (full_motion >= full_active_thr * 0.90 and full_occ >= 0.03)
            or center_motion >= center_active_thr
        )
        if body_signal and angle_in_ranges(full_ang, horizontal_ranges):
            return "PUSHING"
        if body_signal:
            return "PUSHING"
        return "WAITING"

    if cls in crane_classes:
        boom_signal = top60_motion >= (arm_active_thr * 0.52) and top60_occ >= 0.02
        if boom_signal and angle_in_ranges(top60_ang, up_range):
            return "DUMPING"
        if boom_signal:
            return "SWINGING_LOADING"
        return "WAITING"

    if cls in body_only_working_classes:
        body_signal = (
            (full_motion >= full_active_thr * 0.90 and full_occ >= 0.03)
            or center_motion >= center_active_thr
        )
        return "WORKING" if body_signal else "WAITING"

    # Generic fallback
    if motion_source == "arm_only":
        return "DIGGING"
    if motion_source == "both":
        return "SWINGING_LOADING"
    if motion_source == "base_only":
        if "truck" in cls:
            return "DUMPING"
        return "SWINGING_LOADING"
    return "SWINGING_LOADING"


def draw_track_overlay(
    frame: np.ndarray,
    box_xyxy: np.ndarray,
    equipment_id: str,
    equipment_class: str,
    state: str,
    activity: str,
    util_percent: float,
    dwell_s: float,
    total_idle_s: float,
) -> None:
    x1, y1, x2, y2 = box_xyxy.astype(int).tolist()
    color = (30, 200, 30) if state == "ACTIVE" else (40, 70, 240)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = (
        f"{equipment_id} | {equipment_class} | {state} | {activity} | "
        f"util={util_percent:.1f}% | dwell_now={dwell_s:.1f}s | idle_total={total_idle_s:.1f}s"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    pad_x = 4
    pad_y = 4

    # Keep the full label in-frame so all parts (including idle_total) stay highlighted.
    frame_w = frame.shape[1]
    text_x = max(0, min(x1 + 3, frame_w - text_w - (2 * pad_x)))
    rect_x1 = max(0, text_x - pad_x)
    rect_x2 = min(frame_w - 1, text_x + text_w + pad_x)
    rect_y2 = y1
    rect_y1 = max(0, rect_y2 - (text_h + baseline + (2 * pad_y)))

    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
    text_y = rect_y2 - pad_y - baseline
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def run_pipeline(args: argparse.Namespace) -> Dict[str, str]:
    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = os.path.join(args.output_dir, "annotated.mp4")
    out_events_path = os.path.join(args.output_dir, "events.jsonl")
    out_summary_path = os.path.join(args.output_dir, "summary.csv")

    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    timers: Dict[str, EquipmentTimer] = {}
    state_window: Dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=args.state_smooth_window))
    motion_source_window: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=args.state_smooth_window))
    full_motion_window: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=args.motion_smooth_window))
    arm_motion_window: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=args.motion_smooth_window))
    base_motion_window: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=args.motion_smooth_window))
    center_motion_window: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=args.motion_smooth_window))
    last_motion_active_s: Dict[str, float] = {}
    stable_state: Dict[str, str] = {}
    active_votes_needed = min(args.state_active_votes, args.state_smooth_window)
    inactive_votes_needed = min(args.state_inactive_votes, args.state_smooth_window)

    prev_gray: Optional[np.ndarray] = None
    frame_id = 0

    # Lightweight identity memory to reduce ID resets when tracker IDs switch.
    next_equipment_serial = 1
    track_to_equipment: Dict[int, str] = {}
    track_last_seen_s: Dict[int, float] = {}
    entity_last_box: Dict[str, np.ndarray] = {}
    entity_last_seen_s: Dict[str, float] = {}
    entity_last_feature: Dict[str, np.ndarray] = {}
    entity_class: Dict[str, str] = {}
    entity_last_center: Dict[str, Tuple[float, float]] = {}
    entity_velocity: Dict[str, Tuple[float, float]] = {}
    entity_first_seen_s: Dict[str, float] = {}
    entity_hits: Dict[str, int] = defaultdict(int)
    equipment_owner_track: Dict[str, int] = {}
    equipment_owner_seen_s: Dict[str, float] = {}

    with open(out_events_path, "w", encoding="utf-8") as f_events:
        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_s = frame_id / fps
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            global_shift_xy = estimate_global_shift(prev_gray, curr_gray, args.global_motion_scale)

            if args.track_id_ttl_s > 0:
                stale_track_ids = [
                    tid
                    for tid, last_seen_s in track_last_seen_s.items()
                    if (timestamp_s - last_seen_s) > args.track_id_ttl_s
                ]
                for tid in stale_track_ids:
                    track_last_seen_s.pop(tid, None)
                    eq_id = track_to_equipment.pop(tid, None)
                    if eq_id is not None and equipment_owner_track.get(eq_id) == tid:
                        equipment_owner_track.pop(eq_id, None)
                        equipment_owner_seen_s.pop(eq_id, None)

            if frame_id % args.process_every_n_frames == 0:
                result = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    max_det=args.max_det,
                    device=args.device,
                    classes=args.classes,
                    verbose=False,
                )[0]

                detections = []
                if result.boxes is not None and result.boxes.xyxy is not None and len(result.boxes.xyxy) > 0:
                    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
                    boxes_cls = result.boxes.cls.detach().cpu().numpy().astype(int)
                    boxes_conf = result.boxes.conf.detach().cpu().numpy()
                    if result.boxes.id is not None:
                        boxes_id = result.boxes.id.detach().cpu().numpy().astype(int)
                    else:
                        boxes_id = np.arange(len(boxes_xyxy), dtype=int)

                    for det_idx, box_xyxy in enumerate(boxes_xyxy):
                        track_id = int(boxes_id[det_idx])
                        cls_id = int(boxes_cls[det_idx])
                        score = float(boxes_conf[det_idx])
                        cls_name = str(model.names.get(cls_id, f"class_{cls_id}"))

                        if not class_name_allowed(cls_name, args.allowed_class_names):
                            continue

                        x1, y1, x2, y2 = box_xyxy.astype(int).tolist()
                        det_box_area = max(1, x2 - x1) * max(1, y2 - y1)
                        if det_box_area < args.min_box_area:
                            continue

                        detections.append(
                            {
                                "track_id": track_id,
                                "cls_name": cls_name,
                                "score": score,
                                "box_xyxy": box_xyxy,
                                "box_area": det_box_area,
                            }
                        )

                detections = suppress_duplicate_detections(detections, args)

                if args.max_equipment_per_frame > 0 and len(detections) > args.max_equipment_per_frame:
                    detections = sorted(detections, key=lambda d: d["box_area"], reverse=True)[: args.max_equipment_per_frame]

                assigned_in_frame: set[str] = set()
                assigned_boxes_in_frame: Dict[str, np.ndarray] = {}
                assigned_class_in_frame: Dict[str, str] = {}

                for det in detections:
                    track_id = det["track_id"]
                    cls_name = det["cls_name"]
                    score = det["score"]
                    box_xyxy = det["box_xyxy"]

                    roi_feature = extract_appearance_feature(frame, box_xyxy, bins=args.reid_hist_bins)

                    curr_cx, curr_cy = box_center(box_xyxy)
                    frame_diag = float(np.hypot(max(1, width), max(1, height)))

                    # Second-pass same-frame de-duplication against already accepted detections.
                    # This avoids creating a new equipment ID for upper/lower fragments of one machine.
                    skip_as_fragment = False
                    for prev_eq, prev_box in assigned_boxes_in_frame.items():
                        prev_cls = assigned_class_in_frame.get(prev_eq)
                        if prev_cls is not None and prev_cls != cls_name:
                            continue
                        if likely_same_equipment_in_frame(box_xyxy, prev_box, args):
                            skip_as_fragment = True
                            break
                    if skip_as_fragment:
                        continue

                    equipment_id = track_to_equipment.get(track_id)
                    if equipment_id is not None:
                        prev_cls = entity_class.get(equipment_id)
                        if prev_cls is not None and prev_cls != cls_name:
                            track_to_equipment.pop(track_id, None)
                            if equipment_owner_track.get(equipment_id) == track_id:
                                equipment_owner_track.pop(equipment_id, None)
                                equipment_owner_seen_s.pop(equipment_id, None)
                            equipment_id = None
                        else:
                            prev_box = entity_last_box.get(equipment_id)
                            prev_seen_s = entity_last_seen_s.get(equipment_id)
                            prev_center = entity_last_center.get(equipment_id)
                            prev_velocity = entity_velocity.get(equipment_id, (0.0, 0.0))
                            prev_feat = entity_last_feature.get(equipment_id)

                            if prev_box is not None and prev_seen_s is not None and prev_center is not None:
                                gap_s = max(0.0, timestamp_s - prev_seen_s)
                                iou_sticky = bbox_iou(box_xyxy, prev_box)
                                app_sticky = cosine_similarity(roi_feature, prev_feat)
                                area_ratio_sticky = max(box_area(box_xyxy), box_area(prev_box)) / max(
                                    1.0, min(box_area(box_xyxy), box_area(prev_box))
                                )
                                aspect_sticky_curr = box_aspect_ratio(box_xyxy)
                                aspect_sticky_prev = box_aspect_ratio(prev_box)
                                aspect_ratio_sticky = max(aspect_sticky_curr, aspect_sticky_prev) / max(
                                    1e-6, min(aspect_sticky_curr, aspect_sticky_prev)
                                )
                                pred_cx = prev_center[0] + prev_velocity[0] * gap_s
                                pred_cy = prev_center[1] + prev_velocity[1] * gap_s
                                pred_dist_ratio = float(
                                    np.hypot(curr_cx - pred_cx, curr_cy - pred_cy) / max(1.0, frame_diag)
                                )
                                split_like_prev = is_vertical_split_pair(box_xyxy, prev_box, args)

                                # If a tracker ID drifts or is re-used for another machine, detach and rematch.
                                weak_link = (
                                    iou_sticky < args.sticky_min_iou
                                    and app_sticky < args.sticky_min_app
                                    and pred_dist_ratio > args.sticky_max_center_dist_ratio
                                )
                                hard_shape_jump = (
                                    area_ratio_sticky > args.sticky_max_area_ratio
                                    or aspect_ratio_sticky > args.sticky_max_aspect_ratio_ratio
                                ) and not split_like_prev

                                overlap_ambiguous = False
                                for prev_eq, prev_box_in_frame in assigned_boxes_in_frame.items():
                                    if prev_eq == equipment_id:
                                        continue
                                    if assigned_class_in_frame.get(prev_eq) != cls_name:
                                        continue
                                    if (
                                        bbox_iou(box_xyxy, prev_box_in_frame) >= max(0.08, args.dup_iou_thr * 0.35)
                                        and center_distance_ratio(box_xyxy, prev_box_in_frame)
                                        <= (args.dup_center_dist_ratio * 1.35)
                                    ):
                                        overlap_ambiguous = True
                                        break

                                if weak_link and (
                                    hard_shape_jump
                                    or pred_dist_ratio > (args.sticky_max_center_dist_ratio * 1.35)
                                ) and gap_s >= args.sticky_detach_min_gap_s and (not overlap_ambiguous):
                                    track_to_equipment.pop(track_id, None)
                                    if equipment_owner_track.get(equipment_id) == track_id:
                                        equipment_owner_track.pop(equipment_id, None)
                                        equipment_owner_seen_s.pop(equipment_id, None)
                                    equipment_id = None

                    if equipment_id is None:
                        best_match_eq: Optional[str] = None
                        best_match_score = -1.0
                        blocked_by_owner_hold = False

                        for candidate_eq, last_seen_s in entity_last_seen_s.items():
                            gap_s = timestamp_s - last_seen_s
                            if gap_s > args.reid_memory_s:
                                continue
                            if entity_class.get(candidate_eq) != cls_name:
                                continue
                            if candidate_eq in assigned_in_frame:
                                continue

                            owner_tid = equipment_owner_track.get(candidate_eq)
                            if owner_tid is not None and owner_tid != track_id:
                                owner_seen_s = track_last_seen_s.get(owner_tid)
                                if owner_seen_s is not None and (timestamp_s - owner_seen_s) <= args.equipment_owner_hold_s:
                                    blocked_by_owner_hold = True
                                    continue

                            candidate_box = entity_last_box[candidate_eq]
                            iou = bbox_iou(box_xyxy, candidate_box)
                            app_sim = cosine_similarity(roi_feature, entity_last_feature.get(candidate_eq))
                            split_like_candidate = is_vertical_split_pair(box_xyxy, candidate_box, args)
                            cand_area_ratio = max(box_area(box_xyxy), box_area(candidate_box)) / max(
                                1.0, min(box_area(box_xyxy), box_area(candidate_box))
                            )
                            if cand_area_ratio > args.reid_max_area_ratio and not split_like_candidate:
                                continue

                            curr_ar = box_aspect_ratio(box_xyxy)
                            cand_ar = box_aspect_ratio(candidate_box)
                            cand_aspect_ratio = max(curr_ar, cand_ar) / max(1e-6, min(curr_ar, cand_ar))
                            if cand_aspect_ratio > args.reid_max_aspect_ratio_ratio and not split_like_candidate:
                                continue

                            cand_cx, cand_cy = entity_last_center.get(candidate_eq, box_center(candidate_box))
                            vel_x, vel_y = entity_velocity.get(candidate_eq, (0.0, 0.0))
                            pred_cx = cand_cx + vel_x * gap_s
                            pred_cy = cand_cy + vel_y * gap_s
                            center_dist_ratio = float(np.hypot(curr_cx - pred_cx, curr_cy - pred_cy) / max(1.0, frame_diag))

                            if iou < args.reid_min_iou and app_sim < args.reid_min_app and center_dist_ratio > args.reid_max_center_dist_ratio:
                                continue
                            if gap_s > args.reid_max_gap_s and center_dist_ratio > args.reid_long_gap_center_ratio and app_sim < (args.reid_min_app + 0.05):
                                continue

                            center_score = max(0.0, 1.0 - (center_dist_ratio / max(1e-6, args.reid_max_center_dist_ratio)))
                            match_score = (
                                args.reid_iou_weight * iou
                                + args.reid_app_weight * app_sim
                                + args.reid_center_weight * center_score
                            )

                            if match_score < args.reid_min_match_score:
                                continue

                            if match_score > best_match_score:
                                best_match_score = match_score
                                best_match_eq = candidate_eq

                        if best_match_eq is None:
                            if blocked_by_owner_hold:
                                # A likely owner exists but is still in short hold period;
                                # skip this fragment to avoid spawning a new phantom equipment ID.
                                continue
                            if score < args.new_id_min_conf:
                                # Avoid creating short-lived phantom equipment IDs from weak detections.
                                continue
                            equipment_id = f"EQ-{next_equipment_serial:04d}"
                            next_equipment_serial += 1
                        else:
                            equipment_id = best_match_eq

                        track_to_equipment[track_id] = equipment_id

                    if equipment_id in assigned_in_frame:
                        continue
                    assigned_in_frame.add(equipment_id)
                    assigned_boxes_in_frame[equipment_id] = box_xyxy.copy()
                    assigned_class_in_frame[equipment_id] = cls_name

                    equipment_owner_track[equipment_id] = track_id
                    equipment_owner_seen_s[equipment_id] = timestamp_s

                    # FIX6: Detect spatial overlap with already-assigned same-class machines.
                    # During overlap the bounding box is an ambiguous merged region.
                    # Writing it into entity memory would poison subsequent ReID matching
                    # once the machines separate, causing post-overlap ID swaps.
                    spatial_overlap_locked = False
                    for _ov_eq, _ov_box in assigned_boxes_in_frame.items():
                        if _ov_eq == equipment_id:
                            continue
                        if assigned_class_in_frame.get(_ov_eq) != cls_name:
                            continue
                        if (
                            bbox_iou(box_xyxy, _ov_box) >= max(0.10, args.dup_iou_thr * 0.40)
                            or box_containment_ratio(box_xyxy, _ov_box) >= 0.55
                        ):
                            spatial_overlap_locked = True
                            break

                    prev_seen_s = entity_last_seen_s.get(equipment_id)
                    prev_center = entity_last_center.get(equipment_id)

                    curr_diag = max(1.0, box_diag(box_xyxy))
                    center_motion_raw = 0.0
                    if not spatial_overlap_locked:
                        if prev_seen_s is not None and prev_center is not None and timestamp_s > prev_seen_s:
                            dt_center = max(1e-3, timestamp_s - prev_seen_s)
                            disp = float(np.hypot(curr_cx - prev_center[0], curr_cy - prev_center[1]))
                            center_motion_raw = (disp / curr_diag) / dt_center

                            new_vx = (curr_cx - prev_center[0]) / dt_center
                            new_vy = (curr_cy - prev_center[1]) / dt_center
                            old_vx, old_vy = entity_velocity.get(equipment_id, (0.0, 0.0))
                            alpha = 0.6
                            entity_velocity[equipment_id] = (
                                alpha * old_vx + (1.0 - alpha) * new_vx,
                                alpha * old_vy + (1.0 - alpha) * new_vy,
                            )
                        elif equipment_id not in entity_velocity:
                            entity_velocity[equipment_id] = (0.0, 0.0)
                    else:
                        # FIX7: Overlap frame — decay velocity to avoid wrong momentum prediction
                        # after separation, which would guide ReID to the wrong machine.
                        old_vx, old_vy = entity_velocity.get(equipment_id, (0.0, 0.0))
                        entity_velocity[equipment_id] = (old_vx * 0.70, old_vy * 0.70)
                        if equipment_id not in entity_velocity:
                            entity_velocity[equipment_id] = (0.0, 0.0)

                    track_last_seen_s[track_id] = timestamp_s
                    entity_last_seen_s[equipment_id] = timestamp_s
                    entity_class[equipment_id] = cls_name
                    if equipment_id not in entity_first_seen_s:
                        entity_first_seen_s[equipment_id] = timestamp_s

                    if not spatial_overlap_locked:
                        # Only commit clean spatial state when not in an overlap region.
                        entity_last_box[equipment_id] = box_xyxy.copy()
                        entity_last_center[equipment_id] = (curr_cx, curr_cy)
                        if roi_feature is not None:
                            entity_last_feature[equipment_id] = roi_feature

                    entity_hits[equipment_id] += 1
                    if entity_hits[equipment_id] < args.min_track_hits:
                        continue
                    entity_age_s = max(0.0, timestamp_s - entity_first_seen_s.get(equipment_id, timestamp_s))
                    if entity_age_s < args.min_entity_age_s:
                        continue

                    flow_features = compute_motion_scores(
                        prev_gray,
                        curr_gray,
                        box_xyxy,
                        global_shift_xy=global_shift_xy,
                        border_margin=args.motion_border_margin,
                    )
                    full_motion_raw = float(flow_features["full_motion"])
                    arm_motion_raw = float(flow_features["arm_motion"])
                    base_motion_raw = float(flow_features["base_motion"])
                    full_occ_raw = float(flow_features["full_occ"])
                    arm_occ_raw = float(flow_features["arm_occ"])
                    base_occ_raw = float(flow_features["base_occ"])

                    full_motion_window[equipment_id].append(full_motion_raw)
                    arm_motion_window[equipment_id].append(arm_motion_raw)
                    base_motion_window[equipment_id].append(base_motion_raw)
                    center_motion_window[equipment_id].append(center_motion_raw)

                    # Keep ACTIVE responsive to short articulated movement bursts while
                    # still smoothing noise through rolling medians.
                    full_motion = max(
                        full_motion_raw,
                        float(np.median(full_motion_window[equipment_id])),
                    )
                    arm_motion = max(
                        arm_motion_raw,
                        float(np.median(arm_motion_window[equipment_id])),
                    )
                    base_motion = max(
                        base_motion_raw,
                        float(np.median(base_motion_window[equipment_id])),
                    )
                    center_motion = max(
                        center_motion_raw,
                        float(np.median(center_motion_window[equipment_id])),
                    )

                    box_area_px = box_area(box_xyxy)
                    # Large machines need lower occupancy thresholds because true articulated
                    # motion may affect a small fraction of the total box area.
                    occ_scale = float(np.clip(np.sqrt(12000.0 / max(1.0, box_area_px)), 0.45, 1.0))
                    motion_occ_eff = args.min_motion_occupancy * occ_scale
                    part_occ_eff = args.min_part_motion_occupancy * occ_scale

                    instant_state, motion_source = infer_state_and_source(
                        full_motion=full_motion,
                        arm_motion=arm_motion,
                        base_motion=base_motion,
                        center_motion=center_motion,
                        full_occ=full_occ_raw,
                        arm_occ=arm_occ_raw,
                        base_occ=base_occ_raw,
                        full_active_thr=args.full_motion_thr,
                        arm_active_thr=args.arm_motion_thr,
                        base_active_thr=args.base_motion_thr,
                        center_active_thr=args.center_motion_thr,
                        micro_motion_ratio=args.micro_motion_ratio,
                        min_motion_occupancy=motion_occ_eff,
                        min_part_motion_occupancy=part_occ_eff,
                    )

                    # Class-aware ACTIVE boosts for articulated equipment.
                    cls_norm = normalize_label(cls_name)
                    if cls_norm in {"excavator", "backhoe loader"}:
                        arm_angle = float(flow_features.get("arm_angle_deg", float("nan")))
                        arm_signal = arm_motion >= (args.arm_motion_thr * 0.52) and arm_occ_raw >= max(0.02, part_occ_eff * 0.7)
                        if arm_signal and (
                            angle_in_ranges(arm_angle, [(60, 120)])
                            or angle_in_ranges(arm_angle, [(240, 300)])
                            or angle_in_ranges(arm_angle, [(0, 30), (150, 210), (330, 360)])
                        ):
                            instant_state = "ACTIVE"
                            motion_source = "arm_only"
                    elif cls_norm in {"mobile crane", "tower crane"}:
                        top60_motion = float(flow_features.get("top60_motion", 0.0))
                        top60_occ = float(flow_features.get("top60_occ", 0.0))
                        if top60_motion >= (args.arm_motion_thr * 0.52) and top60_occ >= max(0.02, part_occ_eff * 0.7):
                            instant_state = "ACTIVE"
                            motion_source = "arm_only"

                    # Policy requested by product logic:
                    # - Partial motion should become ACTIVE quickly.
                    # - Avoid one-frame jitter toggles to ACTIVE on static machines.
                    any_motion_active = instant_state == "ACTIVE"
                    if any_motion_active:
                        last_motion_active_s[equipment_id] = timestamp_s
                    within_active_hold = (
                        timestamp_s - last_motion_active_s.get(equipment_id, -1e9)
                    ) <= args.active_hold_s

                    strong_motion_active = (
                        (full_motion >= args.full_motion_thr and full_occ_raw >= motion_occ_eff)
                        or (arm_motion >= args.arm_motion_thr and arm_occ_raw >= part_occ_eff)
                        or (base_motion >= args.base_motion_thr and base_occ_raw >= part_occ_eff)
                        or (center_motion >= (args.center_motion_thr * 1.25))
                        or (
                            arm_motion >= (args.arm_motion_thr * 0.92)
                            and arm_occ_raw >= max(0.02, part_occ_eff * 0.80)
                            and (
                                arm_motion >= (base_motion * 1.03)
                                or (arm_motion - base_motion) >= 0.05
                            )
                        )
                        or (
                            base_motion >= (args.base_motion_thr * 0.95)
                            and base_occ_raw >= max(0.02, part_occ_eff * 0.80)
                            and (
                                base_motion >= (arm_motion * 1.03)
                                or (base_motion - arm_motion) >= 0.05
                            )
                        )
                    )

                    state_window[equipment_id].append(any_motion_active)
                    motion_source_window[equipment_id].append(motion_source)
                    active_votes = sum(state_window[equipment_id])
                    inactive_votes = len(state_window[equipment_id]) - active_votes
                    prev_stable_state = stable_state.get(equipment_id, "INACTIVE")

                    if prev_stable_state == "INACTIVE":
                        if strong_motion_active:
                            smoothed_state = "ACTIVE"
                        elif any_motion_active and motion_source in ("arm_only", "base_only", "both"):
                            # FIX5: any partial-motion signal flips to ACTIVE immediately;
                            # active_hold_s keeps it alive across subsequent frames.
                            smoothed_state = "ACTIVE"
                        elif any_motion_active and active_votes >= max(2, active_votes_needed - 1):
                            smoothed_state = "ACTIVE"
                        elif within_active_hold and active_votes >= 1:
                            smoothed_state = "ACTIVE"
                        else:
                            smoothed_state = "INACTIVE"
                    else:
                        if any_motion_active or within_active_hold:
                            smoothed_state = "ACTIVE"
                        else:
                            smoothed_state = "INACTIVE" if inactive_votes >= inactive_votes_needed else "ACTIVE"

                    stable_state[equipment_id] = smoothed_state

                    if smoothed_state == "INACTIVE":
                        smoothed_motion_source = "none"
                    else:
                        non_none_sources = [s for s in motion_source_window[equipment_id] if s != "none"]
                        if non_none_sources:
                            smoothed_motion_source = max(set(non_none_sources), key=non_none_sources.count)
                        elif motion_source != "none":
                            smoothed_motion_source = motion_source
                        else:
                            smoothed_motion_source = "base_only"

                    flow_features_for_activity = dict(flow_features)
                    flow_features_for_activity["full_motion"] = full_motion
                    flow_features_for_activity["arm_motion"] = arm_motion
                    flow_features_for_activity["base_motion"] = base_motion

                    activity = infer_activity(
                        smoothed_state,
                        smoothed_motion_source,
                        cls_name,
                        flow_features_for_activity,
                        args.full_motion_thr,
                        args.arm_motion_thr,
                        args.base_motion_thr,
                        center_motion,
                        args.center_motion_thr,
                    )

                    if smoothed_state == "ACTIVE" and activity == "WAITING":
                        previous_activity = timers[equipment_id].current_activity if equipment_id in timers else "WAITING"
                        if previous_activity != "WAITING":
                            activity = previous_activity
                        else:
                            cls_fallback = normalize_label(cls_name)
                            if cls_fallback in {"dump truck", "wheel loader"}:
                                activity = "MOVING"
                            elif cls_fallback == "dozer":
                                activity = "PUSHING"
                            elif cls_fallback in {"grader", "compactor", "cement truck"}:
                                activity = "WORKING"
                            else:
                                activity = "SWINGING_LOADING"

                    if equipment_id not in timers:
                        timers[equipment_id] = EquipmentTimer(
                            equipment_id=equipment_id,
                            equipment_class=cls_name,
                            last_state=smoothed_state,
                            last_timestamp_s=timestamp_s,
                            inactive_start_s=(timestamp_s if smoothed_state == "INACTIVE" else None),
                        )

                    timers[equipment_id].update(
                        timestamp_s=timestamp_s,
                        state=smoothed_state,
                        activity=activity,
                        motion_source=smoothed_motion_source,
                    )

                    timer = timers[equipment_id]
                    dwell_s = timer.current_dwell_seconds(timestamp_s)

                    event = {
                        "frame_id": frame_id,
                        "track_id": int(track_id),
                        "equipment_id": equipment_id,
                        "equipment_class": cls_name,
                        "timestamp": seconds_to_timestamp_string(timestamp_s),
                        "confidence": round(score, 4),
                        "utilization": {
                            "current_state": smoothed_state,
                            "current_activity": activity,
                            "motion_source": smoothed_motion_source,
                        },
                        "time_analytics": {
                            "total_tracked_seconds": round(timer.total_tracked_seconds, 3),
                            "total_active_seconds": round(timer.total_active_seconds, 3),
                            "total_idle_seconds": round(timer.total_idle_seconds, 3),
                            "utilization_percent": round(timer.utilization_percent, 2),
                            "current_dwell_seconds": round(dwell_s, 3),
                        },
                    }
                    f_events.write(json.dumps(event) + "\n")

                    draw_track_overlay(
                        frame=frame,
                        box_xyxy=box_xyxy,
                        equipment_id=equipment_id,
                        equipment_class=cls_name,
                        state=smoothed_state,
                        activity=activity,
                        util_percent=timer.utilization_percent,
                        dwell_s=dwell_s,
                        total_idle_s=timer.total_idle_seconds,
                    )

            writer.write(frame)
            prev_gray = curr_gray
            frame_id += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    writer.release()

    summary_rows = []
    for eq_id, timer in timers.items():
        if timer.total_tracked_seconds < args.summary_min_tracked_seconds:
            continue
        summary_rows.append(
            {
                "equipment_id": eq_id,
                "equipment_class": timer.equipment_class,
                "total_tracked_seconds": round(timer.total_tracked_seconds, 3),
                "total_active_seconds": round(timer.total_active_seconds, 3),
                "total_idle_seconds": round(timer.total_idle_seconds, 3),
                "utilization_percent": round(timer.utilization_percent, 2),
            }
        )
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("equipment_id")
    else:
        summary_df = pd.DataFrame(
            columns=[
                "equipment_id",
                "equipment_class",
                "total_tracked_seconds",
                "total_active_seconds",
                "total_idle_seconds",
                "utilization_percent",
            ]
        )
    summary_df.to_csv(out_summary_path, index=False)

    return {
        "annotated_video": out_video_path,
        "events_jsonl": out_events_path,
        "summary_csv": out_summary_path,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline pipeline for equipment utilization tracking")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path")
    parser.add_argument("--output_dir", type=str, default="./outputs/baseline_foundation", help="Output folder")
    parser.add_argument("--weights", type=str, default="yolov8x.pt", help="YOLO weights")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Ultralytics tracker config")
    parser.add_argument("--device", type=str, default="0", help="GPU device id or 'cpu'")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum detections per frame")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument(
        "--process_every_n_frames",
        type=int,
        default=1,
        help="Process every Nth frame (1 = every frame)",
    )
    parser.add_argument(
        "--max_equipment_per_frame",
        type=int,
        default=6,
        help="Keep only top-N detections by box area per frame (0 disables)",
    )
    parser.add_argument(
        "--min_track_hits",
        type=int,
        default=12,
        help="Minimum matched frames before a track starts contributing events/timers",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Optional class IDs to keep (depends on your model label map)",
    )
    parser.add_argument(
        "--allowed_class_names",
        type=str,
        nargs="*",
        default=None,
        help="Optional class-name filter for construction equipment",
    )
    parser.add_argument("--min_box_area", type=float, default=2500.0, help="Ignore very small detections")
    parser.add_argument("--full_motion_thr", type=float, default=0.68, help="Full-ROI motion threshold")
    parser.add_argument("--arm_motion_thr", type=float, default=1.2, help="Arm motion threshold")
    parser.add_argument("--base_motion_thr", type=float, default=0.9, help="Base motion threshold")
    parser.add_argument("--center_motion_thr", type=float, default=0.22, help="Center-motion threshold (diag/sec)")
    parser.add_argument(
        "--motion_border_margin",
        type=float,
        default=0.10,
        help="Ignore this fraction of ROI border when computing motion to reduce neighbor contamination",
    )
    parser.add_argument(
        "--min_motion_occupancy",
        type=float,
        default=0.05,
        help="Minimum occupied motion ratio for full-region ACTIVE evidence",
    )
    parser.add_argument(
        "--min_part_motion_occupancy",
        type=float,
        default=0.03,
        help="Minimum occupied motion ratio for arm/base ACTIVE evidence",
    )
    parser.add_argument(
        "--micro_motion_ratio",
        type=float,
        default=0.70,
        help="Fraction of motion thresholds used to promote subtle partial movement to ACTIVE",
    )
    parser.add_argument(
        "--global_motion_scale",
        type=float,
        default=0.35,
        help="Scale used for global camera-motion estimation via phase correlation (0 disables)",
    )
    parser.add_argument("--motion_smooth_window", type=int, default=5, help="Window size for motion smoothing")
    parser.add_argument("--state_smooth_window", type=int, default=7, help="Window size for state smoothing")
    parser.add_argument("--state_active_votes", type=int, default=4, help="ACTIVE votes needed in smoothing window")
    parser.add_argument(
        "--state_inactive_votes",
        type=int,
        default=6,
        help="INACTIVE votes needed in smoothing window when currently ACTIVE (hysteresis)",
    )
    parser.add_argument("--dup_iou_thr", type=float, default=0.35, help="Duplicate suppression IoU threshold")
    parser.add_argument("--dup_containment_thr", type=float, default=0.78, help="Duplicate suppression containment threshold")
    parser.add_argument("--dup_x_overlap_thr", type=float, default=0.68, help="Duplicate suppression x-overlap threshold")
    parser.add_argument("--dup_y_overlap_thr", type=float, default=0.35, help="Duplicate suppression y-overlap threshold")
    parser.add_argument("--dup_center_dist_ratio", type=float, default=0.70, help="Duplicate suppression center-distance ratio")
    parser.add_argument("--dup_max_area_ratio", type=float, default=4.5, help="Duplicate suppression max area ratio")
    parser.add_argument(
        "--dup_vertical_x_overlap_thr",
        type=float,
        default=0.82,
        help="High x-overlap gate for vertical split duplicate suppression",
    )
    parser.add_argument(
        "--dup_vertical_gap_ratio",
        type=float,
        default=0.40,
        help="Max normalized vertical gap for upper/lower split duplicate suppression",
    )
    parser.add_argument(
        "--dup_vertical_center_x_ratio",
        type=float,
        default=0.22,
        help="Max normalized x-center distance for vertical split duplicate suppression",
    )
    parser.add_argument(
        "--dup_vertical_max_area_ratio",
        type=float,
        default=4.2,
        help="Max area ratio for vertical split duplicate suppression",
    )
    parser.add_argument("--track_id_ttl_s", type=float, default=2.0, help="Seconds before stale tracker IDs are dropped")
    parser.add_argument(
        "--new_id_min_conf",
        type=float,
        default=0.0,
        help="Minimum confidence required before creating a brand-new equipment ID",
    )
    parser.add_argument(
        "--min_entity_age_s",
        type=float,
        default=0.0,
        help="Minimum equipment age in seconds before events/timers are emitted",
    )
    parser.add_argument(
        "--equipment_owner_hold_s",
        type=float,
        default=1.2,
        help="When an equipment is recently linked to a track, hold ownership briefly to avoid ID hijack",
    )
    parser.add_argument(
        "--active_hold_s",
        type=float,
        default=0.45,
        help="Hold ACTIVE state briefly after last detected motion to avoid flicker on subtle articulated movement",
    )
    parser.add_argument("--reid_max_gap_s", type=float, default=3.0, help="Maximum gap for reassociation to known equipment")
    parser.add_argument("--reid_memory_s", type=float, default=12.0, help="Maximum memory age for reassociation candidates")
    parser.add_argument("--reid_min_iou", type=float, default=0.25, help="Minimum IoU gate for reassociation")
    parser.add_argument("--reid_min_app", type=float, default=0.78, help="Minimum appearance-similarity gate for reassociation")
    parser.add_argument("--reid_max_center_dist_ratio", type=float, default=0.30, help="Max frame-diagonal ratio for center-distance reassociation gate")
    parser.add_argument("--reid_max_area_ratio", type=float, default=3.2, help="Maximum area ratio gate for reassociation")
    parser.add_argument(
        "--reid_max_aspect_ratio_ratio",
        type=float,
        default=2.2,
        help="Maximum aspect-ratio ratio gate for reassociation",
    )
    parser.add_argument("--reid_long_gap_center_ratio", type=float, default=0.12, help="For long gaps, stricter center-distance ratio gate")
    parser.add_argument("--reid_min_match_score", type=float, default=0.43, help="Minimum weighted reassociation score")
    parser.add_argument("--reid_iou_weight", type=float, default=0.6, help="IoU weight for reassociation scoring")
    parser.add_argument("--reid_app_weight", type=float, default=0.4, help="Appearance weight for reassociation scoring")
    parser.add_argument("--reid_center_weight", type=float, default=0.15, help="Center-distance weight for reassociation scoring")
    parser.add_argument("--reid_hist_bins", type=int, default=16, help="HSV histogram bins per channel for appearance feature")
    parser.add_argument("--sticky_min_iou", type=float, default=0.05, help="Sticky mapping IoU floor before a track->equipment link is considered unstable")
    parser.add_argument("--sticky_min_app", type=float, default=0.62, help="Sticky mapping appearance-similarity floor before a link is considered unstable")
    parser.add_argument("--sticky_max_area_ratio", type=float, default=3.5, help="Sticky mapping max area ratio before forcing rematch")
    parser.add_argument(
        "--sticky_max_aspect_ratio_ratio",
        type=float,
        default=2.4,
        help="Sticky mapping max aspect-ratio ratio before forcing rematch",
    )
    parser.add_argument(
        "--sticky_max_center_dist_ratio",
        type=float,
        default=0.22,
        help="Sticky mapping max frame-diagonal center-distance ratio before forcing rematch",
    )
    parser.add_argument(
        "--sticky_detach_min_gap_s",
        type=float,
        default=0.25,
        help="Minimum time gap before allowing sticky detach to reduce ID swaps during short overlaps",
    )
    parser.add_argument(
        "--summary_min_tracked_seconds",
        type=float,
        default=1.0,
        help="Only include equipment in summary.csv if tracked for at least this many seconds",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.process_every_n_frames < 1:
        raise ValueError("process_every_n_frames must be >= 1")
    if args.max_equipment_per_frame < 0:
        raise ValueError("max_equipment_per_frame must be >= 0")
    if args.min_track_hits < 1:
        raise ValueError("min_track_hits must be >= 1")
    if args.motion_smooth_window < 1:
        raise ValueError("motion_smooth_window must be >= 1")
    if args.full_motion_thr < 0:
        raise ValueError("full_motion_thr must be >= 0")
    if args.state_smooth_window < 1:
        raise ValueError("state_smooth_window must be >= 1")
    if args.state_active_votes < 1:
        raise ValueError("state_active_votes must be >= 1")
    if args.state_inactive_votes < 1:
        raise ValueError("state_inactive_votes must be >= 1")
    if args.center_motion_thr < 0:
        raise ValueError("center_motion_thr must be >= 0")
    if not (0.0 <= args.motion_border_margin < 0.5):
        raise ValueError("motion_border_margin must be in [0, 0.5)")
    if not (0.0 <= args.min_motion_occupancy <= 1.0):
        raise ValueError("min_motion_occupancy must be in [0, 1]")
    if not (0.0 <= args.min_part_motion_occupancy <= 1.0):
        raise ValueError("min_part_motion_occupancy must be in [0, 1]")
    if not (0.0 < args.micro_motion_ratio <= 1.0):
        raise ValueError("micro_motion_ratio must be in (0, 1]")
    if args.global_motion_scale < 0:
        raise ValueError("global_motion_scale must be >= 0")
    if not (0.0 <= args.dup_iou_thr <= 1.0):
        raise ValueError("dup_iou_thr must be in [0, 1]")
    if not (0.0 <= args.dup_containment_thr <= 1.0):
        raise ValueError("dup_containment_thr must be in [0, 1]")
    if not (0.0 <= args.dup_x_overlap_thr <= 1.0):
        raise ValueError("dup_x_overlap_thr must be in [0, 1]")
    if not (0.0 <= args.dup_y_overlap_thr <= 1.0):
        raise ValueError("dup_y_overlap_thr must be in [0, 1]")
    if args.dup_center_dist_ratio < 0:
        raise ValueError("dup_center_dist_ratio must be >= 0")
    if args.dup_max_area_ratio < 1.0:
        raise ValueError("dup_max_area_ratio must be >= 1")
    if not (0.0 <= args.dup_vertical_x_overlap_thr <= 1.0):
        raise ValueError("dup_vertical_x_overlap_thr must be in [0, 1]")
    if args.dup_vertical_gap_ratio < 0:
        raise ValueError("dup_vertical_gap_ratio must be >= 0")
    if args.dup_vertical_center_x_ratio < 0:
        raise ValueError("dup_vertical_center_x_ratio must be >= 0")
    if args.dup_vertical_max_area_ratio < 1.0:
        raise ValueError("dup_vertical_max_area_ratio must be >= 1")
    if args.track_id_ttl_s < 0:
        raise ValueError("track_id_ttl_s must be >= 0")
    if not (0.0 <= args.new_id_min_conf <= 1.0):
        raise ValueError("new_id_min_conf must be in [0, 1]")
    if args.min_entity_age_s < 0:
        raise ValueError("min_entity_age_s must be >= 0")
    if args.equipment_owner_hold_s < 0:
        raise ValueError("equipment_owner_hold_s must be >= 0")
    if args.active_hold_s < 0:
        raise ValueError("active_hold_s must be >= 0")
    if args.reid_max_gap_s < 0:
        raise ValueError("reid_max_gap_s must be >= 0")
    if args.reid_memory_s < 0:
        raise ValueError("reid_memory_s must be >= 0")
    if not (0.0 <= args.reid_min_iou <= 1.0):
        raise ValueError("reid_min_iou must be in [0, 1]")
    if not (0.0 <= args.reid_min_app <= 1.0):
        raise ValueError("reid_min_app must be in [0, 1]")
    if not (0.0 <= args.reid_max_center_dist_ratio <= 1.0):
        raise ValueError("reid_max_center_dist_ratio must be in [0, 1]")
    if args.reid_max_area_ratio < 1.0:
        raise ValueError("reid_max_area_ratio must be >= 1")
    if args.reid_max_aspect_ratio_ratio < 1.0:
        raise ValueError("reid_max_aspect_ratio_ratio must be >= 1")
    if not (0.0 <= args.reid_long_gap_center_ratio <= 1.0):
        raise ValueError("reid_long_gap_center_ratio must be in [0, 1]")
    if not (0.0 <= args.reid_min_match_score <= 1.0):
        raise ValueError("reid_min_match_score must be in [0, 1]")
    if args.reid_hist_bins < 4:
        raise ValueError("reid_hist_bins must be >= 4")
    if not (0.0 <= args.sticky_min_iou <= 1.0):
        raise ValueError("sticky_min_iou must be in [0, 1]")
    if not (0.0 <= args.sticky_min_app <= 1.0):
        raise ValueError("sticky_min_app must be in [0, 1]")
    if args.sticky_max_area_ratio < 1.0:
        raise ValueError("sticky_max_area_ratio must be >= 1")
    if args.sticky_max_aspect_ratio_ratio < 1.0:
        raise ValueError("sticky_max_aspect_ratio_ratio must be >= 1")
    if not (0.0 <= args.sticky_max_center_dist_ratio <= 1.0):
        raise ValueError("sticky_max_center_dist_ratio must be in [0, 1]")
    if args.sticky_detach_min_gap_s < 0:
        raise ValueError("sticky_detach_min_gap_s must be >= 0")
    if args.summary_min_tracked_seconds < 0:
        raise ValueError("summary_min_tracked_seconds must be >= 0")
    if args.reid_memory_s < args.reid_max_gap_s:
        args.reid_memory_s = args.reid_max_gap_s
    if args.state_active_votes > args.state_smooth_window:
        print("state_active_votes is larger than state_smooth_window; clamping to window size.")
        args.state_active_votes = args.state_smooth_window
    if args.state_inactive_votes > args.state_smooth_window:
        print("state_inactive_votes is larger than state_smooth_window; clamping to window size.")
        args.state_inactive_votes = args.state_smooth_window

    weight_sum = args.reid_iou_weight + args.reid_app_weight + args.reid_center_weight
    if weight_sum <= 0:
        raise ValueError("reid_iou_weight + reid_app_weight + reid_center_weight must be > 0")
    args.reid_iou_weight = args.reid_iou_weight / weight_sum
    args.reid_app_weight = args.reid_app_weight / weight_sum
    args.reid_center_weight = args.reid_center_weight / weight_sum

    outputs = run_pipeline(args)
    print("\nBaseline pipeline completed.")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
