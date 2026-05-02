#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np

import auto_solver as solver


def parse_grid_from_txt(path: Path) -> np.ndarray:
    raw = path.read_text(encoding="utf-8")
    rows = []
    for line in raw.splitlines():
        nums = re.findall(r"-?\d+", line)
        if nums:
            rows.append([int(x) for x in nums])
    if not rows:
        raise ValueError(f"No numbers found in {path}")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("Grid rows have inconsistent lengths")
    return np.array(rows, dtype=np.int32)


def enumerate_solutions(grid: np.ndarray, conf: np.ndarray, min_cell_conf: float):
    rows, cols = grid.shape
    out = []
    for r1 in range(rows):
        for r2 in range(r1, rows):
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    sub = grid[r1 : r2 + 1, c1 : c2 + 1]
                    if np.any(sub < 0):
                        continue
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    s = int(np.sum(sub))
                    score = float(np.mean(conf[r1 : r2 + 1, c1 : c2 + 1]))
                    if s == 10 and score >= min_cell_conf:
                        points = int(np.sum(sub > 0))
                        out.append((r1, c1, r2, c2, area, s, score, points))
    out.sort(key=lambda x: (-x[4], -x[6], x[0], x[1], x[2], x[3]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Test solver on test.txt-like grid file")
    parser.add_argument("--file", default="test.txt", help="Path to txt grid file")
    parser.add_argument("--min-cell-conf", type=float, default=0.45)
    parser.add_argument("--solver-backend", choices=["local", "model"], default="local")
    parser.add_argument("--llm-provider", choices=["mimo", "openai"], default="mimo")
    parser.add_argument("--openai-model", default="mimo-v2.5-pro")
    parser.add_argument("--model-max-candidates", type=int, default=200)
    parser.add_argument("--model-strength-test", action="store_true", help="Evaluate model against local full solution set")
    parser.add_argument("--model-strength-runs", type=int, default=5, help="How many model runs in strength test")
    parser.add_argument("--fallback-local-on-none", action="store_true", help="Model returns None -> fallback to local solver")
    parser.add_argument("--show-all", action="store_true", help="Print all valid solutions")
    parser.add_argument("--simulate-until-no-solution", action="store_true", help="Local greedy simulate and output total points")
    parser.add_argument("--log-file", default="test_txt_solver.log", help="Write test result log")
    args = parser.parse_args()

    log_fp = open(args.log_file, "w", encoding="utf-8") if args.log_file else None

    def log(*parts):
        msg = " ".join(str(x) for x in parts)
        print(msg)
        if log_fp is not None:
            log_fp.write(msg + "\n")
            log_fp.flush()

    grid = parse_grid_from_txt(Path(args.file))
    conf = np.ones_like(grid, dtype=np.float32)

    log(f"grid_shape={grid.shape}")
    log("grid:")
    log(grid)

    model_moves = []
    if args.solver_backend == "model":
        model_moves = solver.find_moves_sum10_openai(
            grid,
            conf,
            args.min_cell_conf,
            args.llm_provider,
            args.openai_model,
            args.model_max_candidates,
        )
        best = model_moves[0] if model_moves else None
        log("backend=model")
        log(f"model_valid_candidates={len(model_moves)}")
        if best is None and args.fallback_local_on_none:
            log("model_best=None, fallback_to_local=True")
            best = solver.find_best_rect_sum_10(grid, conf, args.min_cell_conf)
            log("fallback_backend=local")
    else:
        best = solver.find_best_rect_sum_10(grid, conf, args.min_cell_conf)
        log("backend=local")

    if best is None:
        log("best=None")
    else:
        sub = grid[best.r1 : best.r2 + 1, best.c1 : best.c2 + 1]
        s = int(np.sum(sub))
        gained = int(np.sum(sub > 0))
        log(
            f"best=({best.r1},{best.c1})->({best.r2},{best.c2}) "
            f"area={best.area} sum={s} score={best.score:.3f} points={gained}"
        )

    if args.show_all:
        all_solutions = enumerate_solutions(grid, conf, args.min_cell_conf)
        log(f"all_valid_count={len(all_solutions)}")
        for i, (r1, c1, r2, c2, area, s, score, points) in enumerate(all_solutions, 1):
            log(
                f"{i:03d}. ({r1},{c1})->({r2},{c2}) "
                f"area={area} sum={s} score={score:.3f} points={points}"
            )

    if args.model_strength_test:
        all_solutions = enumerate_solutions(grid, conf, args.min_cell_conf)
        local_set = {(r1, c1, r2, c2) for (r1, c1, r2, c2, *_rest) in all_solutions}
        local_best = all_solutions[0] if all_solutions else None
        local_best_key = (local_best[0], local_best[1], local_best[2], local_best[3]) if local_best else None
        log("=== model_strength_test ===")
        log(f"local_total_solutions={len(local_set)}")
        if local_best_key is not None:
            log(f"local_best={local_best_key}")
        for run_idx in range(1, args.model_strength_runs + 1):
            try:
                moves = solver.find_moves_sum10_openai(
                    grid,
                    conf,
                    args.min_cell_conf,
                    args.llm_provider,
                    args.openai_model,
                    args.model_max_candidates,
                )
                model_keys = {(m.r1, m.c1, m.r2, m.c2) for m in moves}
                overlap = len(model_keys & local_set)
                coverage = (overlap / len(local_set) * 100.0) if local_set else 0.0
                precision = (overlap / len(model_keys) * 100.0) if model_keys else 0.0
                hit_best = local_best_key in model_keys if local_best_key else False
                log(
                    f"run={run_idx} model_candidates={len(model_keys)} overlap={overlap} "
                    f"coverage={coverage:.1f}% precision={precision:.1f}% hit_local_best={hit_best}"
                )
                if moves:
                    m0 = moves[0]
                    log(f"run={run_idx} model_top1=({m0.r1},{m0.c1})->({m0.r2},{m0.c2}) area={m0.area}")
            except Exception as e:
                log(f"run={run_idx} model_error={type(e).__name__}: {e}")

    if args.simulate_until_no_solution:
        if args.solver_backend != "local":
            log("simulate_until_no_solution requires --solver-backend local")
        else:
            sim_grid = grid.copy()
            sim_conf = conf.copy()
            total_points = 0
            step = 0
            while True:
                mv = solver.find_best_rect_sum_10(sim_grid, sim_conf, args.min_cell_conf)
                if mv is None:
                    break
                sub = sim_grid[mv.r1 : mv.r2 + 1, mv.c1 : mv.c2 + 1]
                gained = int(np.sum(sub > 0))
                s = int(np.sum(sub))
                total_points += gained
                step += 1
                log(
                    f"sim_step={step} move=({mv.r1},{mv.c1})->({mv.r2},{mv.c2}) "
                    f"sum={s} area={mv.area} points={gained} total_points={total_points}"
                )
                sim_grid[mv.r1 : mv.r2 + 1, mv.c1 : mv.c2 + 1] = 0
                sim_conf[mv.r1 : mv.r2 + 1, mv.c1 : mv.c2 + 1] = 1.0
            log(f"sim_end steps={step} total_points={total_points}")

    if log_fp is not None:
        log_fp.close()


if __name__ == "__main__":
    main()
