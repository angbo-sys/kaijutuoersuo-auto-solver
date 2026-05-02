#!/usr/bin/env python3
import argparse
import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import ImageGrab
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


_OPENAI_CLIENT = None

# =========================
# LLM API Config (edit here)
# =========================
MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_API_KEY = ""
OPENAI_API_KEY = ""
MIMO_VISION_MODEL = "mimo-v2.5"


@dataclass
class Rect:
    r1: int
    c1: int
    r2: int
    c2: int
    area: int
    score: float


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Please run: pip install openai")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is empty in auto_solver.py")
        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _OPENAI_CLIENT


def _build_system_prompt() -> str:
    now = datetime.now()
    date_s = now.strftime("%Y-%m-%d")
    weekday_s = now.strftime("%A")
    return (
        "你是MiMo（中文名称也是MiMo），是小米公司研发的AI智能助手。"
        f"今天的日期：{date_s} {weekday_s}，你的知识截止日期是2024年12月。"
    )


def _get_solver_client(provider: str):
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please run: pip install openai")
    if provider == "mimo":
        if not MIMO_API_KEY:
            raise RuntimeError("MIMO_API_KEY is empty in auto_solver.py")
        return OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
    return _get_openai_client()


def ask_point(prompt: str) -> Tuple[int, int]:
    input(prompt)
    p = pyautogui.position()
    return p.x, p.y


def capture_board(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    img = ImageGrab.grab(bbox=(left, top, right, bottom))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess_cell(cell_bgr: np.ndarray, mode: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if mode == 0:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            6,
        )

    # Ensure black digit on white background for OCR stability.
    black_ratio = np.mean(th < 128)
    if black_ratio > 0.5:
        th = 255 - th

    th = cv2.copyMakeBorder(th, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    th = cv2.resize(th, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    return th


def _ocr_once(img: np.ndarray) -> Tuple[int, float]:
    config = "--oem 3 --psm 10 -c tessedit_char_whitelist=123456789"
    data = pytesseract.image_to_data(
        img,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    best_digit = -1
    best_conf = -1.0
    for txt, conf_s in zip(data["text"], data["conf"]):
        txt = (txt or "").strip()
        if len(txt) != 1 or txt < "1" or txt > "9":
            continue
        try:
            conf = float(conf_s)
        except (TypeError, ValueError):
            conf = -1.0
        if conf > best_conf:
            best_conf = conf
            best_digit = int(txt)
    if best_digit == -1:
        return -1, 0.0
    return best_digit, max(0.0, min(best_conf / 100.0, 1.0))


def ocr_digit(cell_bgr: np.ndarray) -> Tuple[int, float]:
    votes = {}
    conf_sum = {}
    for mode in (0, 1):
        img = preprocess_cell(cell_bgr, mode=mode)
        d, c = _ocr_once(img)
        if d == -1:
            continue
        votes[d] = votes.get(d, 0) + 1
        conf_sum[d] = conf_sum.get(d, 0.0) + c
    if not votes:
        return -1, 0.0
    best = sorted(votes.keys(), key=lambda k: (votes[k], conf_sum[k]), reverse=True)[0]
    avg_conf = conf_sum[best] / votes[best]
    return best, avg_conf


def split_and_ocr(board: np.ndarray, rows: int, cols: int, margin_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = board.shape[:2]
    ch = h / rows
    cw = w / cols
    out = np.zeros((rows, cols), dtype=np.int32)
    conf = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            y1 = int(r * ch)
            y2 = int((r + 1) * ch)
            x1 = int(c * cw)
            x2 = int((c + 1) * cw)

            mx = int((x2 - x1) * margin_ratio)
            my = int((y2 - y1) * margin_ratio)
            x1c = min(x2 - 1, x1 + mx)
            x2c = max(x1c + 1, x2 - mx)
            y1c = min(y2 - 1, y1 + my)
            y2c = max(y1c + 1, y2 - my)

            cell = board[y1c:y2c, x1c:x2c]
            d, cf = ocr_digit(cell)
            out[r, c] = d
            conf[r, c] = cf
    return out, conf


def split_and_ocr_by_model(board: np.ndarray, rows: int, cols: int, model: str) -> Tuple[np.ndarray, np.ndarray]:
    ok, buf = cv2.imencode(".png", board)
    if not ok:
        raise RuntimeError("Failed to encode board image.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    prompt = (
        f"Read the number grid image and return compact JSON only with exact shape {rows}x{cols}. "
        'Schema: {"grid":[[n,...],[...]]}. '
        "Each n must be integer 1-9 if readable, otherwise -1. "
        "Do not output analysis, do not use markdown fences, and keep output as a single JSON object."
    )
    client = _get_solver_client("mimo")
    def _call_once(max_tokens: int):
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            max_completion_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
        )

    resp = _call_once(2000)
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if choice.finish_reason == "length" or not text:
        resp = _call_once(5000)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            reasoning = getattr(choice.message, "reasoning_content", None)
            if reasoning:
                text = reasoning.strip()
    if not text:
        raise RuntimeError("Model OCR returned empty text.")
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    grid = data.get("grid", [])
    if len(grid) != rows or any(not isinstance(r, list) or len(r) != cols for r in grid):
        raise RuntimeError("Model OCR returned invalid grid shape.")

    out = np.zeros((rows, cols), dtype=np.int32)
    conf = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            try:
                d = int(grid[r][c])
            except Exception:
                d = -1
            if 1 <= d <= 9:
                out[r, c] = d
                conf[r, c] = 1.0
            else:
                out[r, c] = -1
                conf[r, c] = 0.0
    return out, conf


def find_best_rect_sum_10(grid: np.ndarray, conf: np.ndarray, min_cell_conf: float) -> Optional[Rect]:
    rows, cols = grid.shape
    best: Optional[Rect] = None

    # Correctness-first solver:
    # Fix top/bottom rows, compress columns, then exhaustively check all column spans.
    # For this board size (16x10), this is still very fast and avoids missed candidates.
    for r1 in range(rows):
        col_sum = np.zeros(cols, dtype=np.int32)
        col_known = np.zeros(cols, dtype=np.int32)
        col_conf = np.zeros(cols, dtype=np.float32)
        for r2 in range(r1, rows):
            col_sum += grid[r2, :]
            # -1 means unknown/unreadable; 0 means cleared/empty and is selectable.
            col_known += (grid[r2, :] >= 0).astype(np.int32)
            col_conf += conf[r2, :]

            ps_sum = np.zeros(cols + 1, dtype=np.int32)
            ps_known = np.zeros(cols + 1, dtype=np.int32)
            ps_conf = np.zeros(cols + 1, dtype=np.float32)
            ps_sum[1:] = np.cumsum(col_sum)
            ps_known[1:] = np.cumsum(col_known)
            ps_conf[1:] = np.cumsum(col_conf)

            height = r2 - r1 + 1
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    seg_sum = int(ps_sum[c2 + 1] - ps_sum[c1])
                    if seg_sum != 10:
                        continue

                    width = c2 - c1 + 1
                    area = height * width
                    known_cnt = int(ps_known[c2 + 1] - ps_known[c1])
                    if known_cnt != area:
                        continue

                    conf_sum = float(ps_conf[c2 + 1] - ps_conf[c1])
                    avg_conf = conf_sum / area
                    if avg_conf < min_cell_conf:
                        continue

                    cand = Rect(r1, c1, r2, c2, area, avg_conf)
                    if best is None:
                        best = cand
                    else:
                        # Sum==10 is hard constraint.
                        # Prefer eliminating more cells first (area), then OCR confidence.
                        if cand.area > best.area:
                            best = cand
                        elif cand.area == best.area and cand.score > best.score + 1e-6:
                            best = cand
                        elif cand.area == best.area and abs(cand.score - best.score) <= 1e-6:
                            if (cand.r1, cand.c1, cand.r2, cand.c2) < (best.r1, best.c1, best.r2, best.c2):
                                best = cand
    return best


def _validate_and_rank_moves(
    grid: np.ndarray, conf: np.ndarray, min_cell_conf: float, moves_raw: list
) -> list:
    rows, cols = grid.shape
    valid = []
    seen = set()
    for mv in moves_raw:
        try:
            r1 = int(mv["r1"])
            c1 = int(mv["c1"])
            r2 = int(mv["r2"])
            c2 = int(mv["c2"])
        except Exception:
            continue
        key = (r1, c1, r2, c2)
        if key in seen:
            continue
        seen.add(key)
        if r1 > r2 or c1 > c2:
            continue
        if not (0 <= r1 < rows and 0 <= r2 < rows and 0 <= c1 < cols and 0 <= c2 < cols):
            continue
        sub_grid = grid[r1 : r2 + 1, c1 : c2 + 1]
        sub_conf = conf[r1 : r2 + 1, c1 : c2 + 1]
        if np.any(sub_grid < 0):
            continue
        score = float(np.mean(sub_conf))
        if score < min_cell_conf:
            continue
        s = int(np.sum(sub_grid))
        if s != 10:
            continue
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        valid.append(Rect(r1=r1, c1=c1, r2=r2, c2=c2, area=area, score=score))
    valid.sort(key=lambda x: (-x.area, -x.score, x.r1, x.c1, x.r2, x.c2))
    return valid


def find_moves_sum10_openai(
    grid: np.ndarray,
    conf: np.ndarray,
    min_cell_conf: float,
    provider: str,
    model: str,
    max_candidates: int,
) -> list:
    payload = {
        "rows": int(grid.shape[0]),
        "cols": int(grid.shape[1]),
        "min_cell_conf": float(min_cell_conf),
        "grid": grid.tolist(),
        "conf": np.round(conf, 4).tolist(),
    }
    prompt = (
        "你在解一个数字消除游戏。规则：框选一个轴对齐矩形，若矩形内数字和=10，则该矩形会被消除。\n"
        "请基于当前识别结果输出尽可能完整的可行题解列表。\n\n"
        "硬约束：\n"
        "1) 矩形必须在棋盘内，且 r1<=r2, c1<=c2。\n"
        "2) 仅能使用已识别格子（grid>=0），不能包含 -1。注意：0 表示已消除空格，可被横跨框选。\n"
        "3) 所选矩形的平均置信度必须 >= min_cell_conf。\n"
        "4) 所选矩形内数字总和必须严格等于 10。\n\n"
        "优化目标（按优先级）：\n"
        "A. 优先消除更多格子。\n"
        "B. 面积相同则平均置信度更高。\n"
        "C. 仍相同则按 (r1,c1,r2,c2) 字典序最小。\n\n"
        "只输出 JSON，不要 markdown，不要解释。\n"
        f'格式: {{"moves":[{{"r1":int,"c1":int,"r2":int,"c2":int}}, ...], "count_estimate":int}}。'
        f"moves 按优化目标排序，最多输出 {max_candidates} 条。\n\n"
        f"输入数据:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    client = _get_solver_client(provider)
    def _call_once(max_tokens: int):
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
        )

    resp = _call_once(800)
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if choice.finish_reason == "length" or not text:
        resp = _call_once(1800)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
    if not text:
        return []
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    moves_raw = data.get("moves")
    if not isinstance(moves_raw, list):
        move = data.get("move")
        moves_raw = [move] if isinstance(move, dict) else []
    valid = _validate_and_rank_moves(grid, conf, min_cell_conf, moves_raw)
    return valid


def find_best_rect_sum_10_openai(
    grid: np.ndarray,
    conf: np.ndarray,
    min_cell_conf: float,
    provider: str,
    model: str,
    max_candidates: int,
) -> Optional[Rect]:
    moves = find_moves_sum10_openai(
        grid, conf, min_cell_conf, provider, model, max_candidates
    )
    return moves[0] if moves else None


def cell_corners(board_lt: Tuple[int, int], board_rb: Tuple[int, int], rows: int, cols: int, r: int, c: int) -> Tuple[float, float, float, float]:
    x1, y1 = board_lt
    x2, y2 = board_rb
    cw = (x2 - x1) / cols
    ch = (y2 - y1) / rows
    left = x1 + c * cw
    top = y1 + r * ch
    right = x1 + (c + 1) * cw
    bottom = y1 + (r + 1) * ch
    return left, top, right, bottom


def drag_rect(
    board_lt: Tuple[int, int],
    board_rb: Tuple[int, int],
    rows: int,
    cols: int,
    rect: Rect,
    drag_duration: float,
    edge_inset_ratio: float,
    mouse_settle: float,
    down_settle: float,
    drag_end_settle: float,
    distance_settle_scale: float,
):
    tl_left, tl_top, tl_right, tl_bottom = cell_corners(board_lt, board_rb, rows, cols, rect.r1, rect.c1)
    br_left, br_top, br_right, br_bottom = cell_corners(board_lt, board_rb, rows, cols, rect.r2, rect.c2)

    cell_w = (board_rb[0] - board_lt[0]) / cols
    cell_h = (board_rb[1] - board_lt[1]) / rows
    inset_x = max(2.0, cell_w * edge_inset_ratio)
    inset_y = max(2.0, cell_h * edge_inset_ratio)

    # Drag from rectangle inner top-left corner to inner bottom-right corner.
    sx = int(tl_left + inset_x)
    sy = int(tl_top + inset_y)
    ex = int(br_right - inset_x)
    ey = int(br_bottom - inset_y)

    # Fallback guard for tiny cells / extreme insets.
    if ex <= sx:
        sx = int((tl_left + tl_right) * 0.5)
        ex = int((br_left + br_right) * 0.5)
    if ey <= sy:
        sy = int((tl_top + tl_bottom) * 0.5)
        ey = int((br_top + br_bottom) * 0.5)

    drag_dist = float(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5)
    extra_settle = drag_dist * max(0.0, distance_settle_scale)

    pyautogui.moveTo(sx, sy, duration=0.05)
    if mouse_settle > 0:
        time.sleep(mouse_settle)
    pyautogui.mouseDown(button="left")
    hold_before_drag = down_settle + extra_settle
    if hold_before_drag > 0:
        time.sleep(hold_before_drag)
    pyautogui.moveTo(ex, ey, duration=drag_duration)
    if drag_end_settle > 0:
        time.sleep(drag_end_settle)
    pyautogui.mouseUp(button="left")


def main():
    parser = argparse.ArgumentParser(description="微信小游戏和为10矩形自动求解")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--loops", type=int, default=0, help="最大执行次数；0表示不限次数，直到无解")
    parser.add_argument("--interval", type=float, default=0.25, help="每次识别间隔秒")
    parser.add_argument("--drag-duration", type=float, default=0.16)
    parser.add_argument("--mouse-settle", type=float, default=0.03, help="拖拽前鼠标落点停顿秒数")
    parser.add_argument("--down-settle", type=float, default=0.03, help="按下鼠标后停顿秒数")
    parser.add_argument("--drag-end-settle", type=float, default=0.05, help="到达终点后松开前停顿秒数")
    parser.add_argument("--distance-settle-scale", type=float, default=0.00015, help="按拖拽距离增加停顿秒数(秒/像素)")
    parser.add_argument("--edge-inset-ratio", type=float, default=0.18, help="拖拽起止点距离格子边缘的内缩比例")
    parser.add_argument("--margin-ratio", type=float, default=0.16, help="单元格裁剪边距比例")
    parser.add_argument("--min-cell-conf", type=float, default=0.45, help="单元格最小OCR置信度(0-1)")
    parser.add_argument("--ocr-backend", choices=["tesseract", "model"], default="model", help="数字识别后端")
    parser.add_argument("--vision-model", type=str, default=MIMO_VISION_MODEL, help="模型OCR使用的视觉模型名")
    parser.add_argument("--ocr-every-n", type=int, default=1, help="每N轮做一次OCR识别；中间轮次用本地推演")
    parser.add_argument("--ocr-once", action="store_true", help="只在第一轮做一次OCR，后续仅本地推演")
    parser.add_argument("--reocr-on-no-solution", action="store_true", help="无解时立即重新OCR并进入新一轮解题")
    parser.add_argument("--solver-backend", choices=["openai", "local"], default="local", help="求解后端：模型 or 本地")
    parser.add_argument("--llm-provider", choices=["mimo", "openai"], default="mimo", help="模型服务商，默认小米MiMo")
    parser.add_argument("--openai-model", type=str, default="mimo-v2.5-pro", help="模型名，仅在 --solver-backend=openai 生效")
    parser.add_argument("--model-max-candidates", type=int, default=200, help="模型最多返回候选解条数")
    parser.add_argument("--dry-run", action="store_true", help="只识别和求解，不拖拽")
    parser.add_argument("--log-file", type=str, default="", help="将运行日志追加写入文件")
    args = parser.parse_args()

    log_fp = open(args.log_file, "a", encoding="utf-8") if args.log_file else None

    def log(*parts):
        msg = " ".join(str(x) for x in parts)
        print(msg)
        if log_fp is not None:
            log_fp.write(msg + "\n")
            log_fp.flush()

    log("3秒后请切到微信游戏窗口...")
    time.sleep(3)

    lt = ask_point("把鼠标移到数字棋盘左上角(不含边框)，回车记录... ")
    rb = ask_point("把鼠标移到数字棋盘右下角(不含边框)，回车记录... ")
    log(f"棋盘区域: {lt} -> {rb}")

    if args.ocr_every_n <= 0:
        raise ValueError("--ocr-every-n 必须 >= 1")

    cached_grid: Optional[np.ndarray] = None
    cached_conf: Optional[np.ndarray] = None
    rounds_since_ocr = args.ocr_every_n
    total_points = 0
    cycle_id = 1

    max_loops = args.loops if args.loops > 0 else 10**9
    for i in range(max_loops):
        need_ocr = (
            cached_grid is None
            or cached_conf is None
            or (not args.ocr_once and rounds_since_ocr >= args.ocr_every_n)
        )

        if need_ocr:
            board = capture_board(lt[0], lt[1], rb[0], rb[1])
            if args.ocr_backend == "model":
                try:
                    grid, conf = split_and_ocr_by_model(board, args.rows, args.cols, args.vision_model)
                except Exception as e:
                    log(f"模型OCR失败({type(e).__name__}): {e}")
                    if cached_grid is not None and cached_conf is not None:
                        log("使用上一次OCR结果继续。")
                        grid = cached_grid.copy()
                        conf = cached_conf.copy()
                    else:
                        log("无可用缓存，回退 tesseract OCR。")
                        grid, conf = split_and_ocr(board, args.rows, args.cols, args.margin_ratio)
            else:
                grid, conf = split_and_ocr(board, args.rows, args.cols, args.margin_ratio)
            cached_grid = grid.copy()
            cached_conf = conf.copy()
            rounds_since_ocr = 0
            source = "OCR"
        else:
            grid = cached_grid.copy()
            conf = cached_conf.copy()
            source = "SIM"

        if args.solver_backend == "openai":
            try:
                best = find_best_rect_sum_10_openai(
                    grid, conf, args.min_cell_conf, args.llm_provider, args.openai_model, args.model_max_candidates
                )
            except Exception as e:
                log(f"模型求解失败，回退本地求解: {e}")
                best = find_best_rect_sum_10(grid, conf, args.min_cell_conf)
        else:
            best = find_best_rect_sum_10(grid, conf, args.min_cell_conf)

        loop_total = "∞" if args.loops == 0 else str(args.loops)
        log(
            f"[{i+1}/{loop_total}] cycle={cycle_id} source={source} "
            f"ocr_every_n={args.ocr_every_n} since_ocr={rounds_since_ocr}"
        )
        log(grid)
        log("avg_conf=", round(float(np.mean(conf)), 3), "total_points=", total_points)

        if best is None:
            # If no solution appears during SIM, allow one OCR refresh attempt.
            # If still no solution on fresh OCR frame, stop.
            if args.reocr_on_no_solution and source != "OCR":
                log("未找到和为10的矩形，重新识别一次后再判断。")
                cached_grid = None
                cached_conf = None
                rounds_since_ocr = args.ocr_every_n
                cycle_id += 1
                time.sleep(args.interval)
                continue
            log("未找到和为10的矩形，停止。")
            break

        sub = grid[best.r1:best.r2 + 1, best.c1:best.c2 + 1]
        s = int(sub.sum())
        gained = int(np.sum(sub > 0))
        total_points += gained
        log(
            f"选择: ({best.r1},{best.c1}) -> ({best.r2},{best.c2}), "
            f"area={best.area}, sum={s}, score={best.score:.3f}, "
            f"gained_points={gained}, total_points={total_points}"
        )

        if not args.dry_run:
            drag_rect(
                lt,
                rb,
                args.rows,
                args.cols,
                best,
                args.drag_duration,
                args.edge_inset_ratio,
                args.mouse_settle,
                args.down_settle,
                args.drag_end_settle,
                args.distance_settle_scale,
            )

        # 本地推演：把已消除区域标记为 0（空格，可横跨）。
        if cached_grid is not None and cached_conf is not None:
            cached_grid[best.r1:best.r2 + 1, best.c1:best.c2 + 1] = 0
            cached_conf[best.r1:best.r2 + 1, best.c1:best.c2 + 1] = 1.0
        rounds_since_ocr += 1

        time.sleep(args.interval)

    log(f"结束，总分={total_points}")
    if log_fp is not None:
        log_fp.close()


if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    main()
