import re
import json
import numpy as np
import importlib.util
from pathlib import Path
from datetime import datetime

# Load solver module from source file
spec = importlib.util.spec_from_file_location('auto_solver_src', '/Users/yelainab/project/wx_game/auto_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Parse test.txt
raw = Path('/Users/yelainab/project/wx_game/test.txt').read_text(encoding='utf-8')
rows = []
for line in raw.splitlines():
    nums = re.findall(r'-?\d+', line)
    if nums:
        rows.append([int(x) for x in nums])

grid = np.array(rows, dtype=np.int32)
conf = np.ones_like(grid, dtype=np.float32)
min_conf = 0.45

print('=== META ===')
print('time=', datetime.now().isoformat())
print('grid_shape=', grid.shape)
print('min_conf=', min_conf)
print('mimo_base_url=', getattr(mod, 'MIMO_BASE_URL', ''))
print('mimo_key_set=', bool(getattr(mod, 'MIMO_API_KEY', '')))
print('')

print('=== GRID ===')
for r in range(grid.shape[0]):
    print(' '.join(f'{int(x):2d}' for x in grid[r]))
print('')

# Enumerate all valid rectangles (complete solution set for static board)
R, C = grid.shape
cands = []
for r1 in range(R):
    for r2 in range(r1, R):
        for c1 in range(C):
            for c2 in range(c1, C):
                sub = grid[r1:r2+1, c1:c2+1]
                if np.any(sub <= 0):
                    continue
                s = int(np.sum(sub))
                if s != 10:
                    continue
                area = (r2-r1+1)*(c2-c1+1)
                avg = float(np.mean(conf[r1:r2+1, c1:c2+1]))
                if avg < min_conf:
                    continue
                cands.append((r1, c1, r2, c2, area, s, avg))

# Sort by rule: area desc, conf desc, lexicographically asc
cands_sorted = sorted(cands, key=lambda x: (-x[4], -x[6], x[0], x[1], x[2], x[3]))

print('=== LOCAL ENUM LOG ===')
print('valid_rectangles_count=', len(cands_sorted))
print('top_20_by_rule:')
for i, (r1,c1,r2,c2,area,s,avg) in enumerate(cands_sorted[:20], 1):
    print(f'{i:02d}. ({r1},{c1})->({r2},{c2}) area={area} sum={s} conf={avg:.3f}')
print('')

if cands_sorted:
    br1, bc1, br2, bc2, barea, bs, bavg = cands_sorted[0]
    print('LOCAL_BEST=', {'r1':br1,'c1':bc1,'r2':br2,'c2':bc2,'area':barea,'sum':bs,'score':round(bavg,3)})
else:
    print('LOCAL_BEST=None')
print('')

print('=== MODEL SOLVE LOG ===')
payload = {
    'rows': int(grid.shape[0]),
    'cols': int(grid.shape[1]),
    'min_cell_conf': float(min_conf),
    'grid': grid.tolist(),
    'conf': np.round(conf, 4).tolist(),
}
prompt = (
    '你在解一个数字消除游戏。规则：框选一个轴对齐矩形，若矩形内数字和=10，则该矩形会被消除。\\n'
    '请基于当前识别结果选择下一步最优框选。\\n'
    '硬约束：矩形在棋盘内；不能包含-1；平均置信度>=min_cell_conf；和必须=10。\\n'
    '优化目标：先面积最大，再置信度，再字典序。\\n'
    '只输出JSON：{"move":{"r1":int,"c1":int,"r2":int,"c2":int}} 或 {"move":null}。\\n'
    f'输入: {json.dumps(payload, ensure_ascii=False)}'
)

try:
    client = mod._get_solver_client('mimo')
    resp = client.chat.completions.create(
        model='mimo-v2.5-pro',
        messages=[
            {'role':'system','content':mod._build_system_prompt()},
            {'role':'user','content':prompt},
        ],
        max_completion_tokens=5000,
        temperature=0.1,
        top_p=0.95,
    )
    ch = resp.choices[0]
    print('finish_reason=', ch.finish_reason)
    print('content_repr=', repr(ch.message.content))
    reasoning = getattr(ch.message, 'reasoning_content', None)
    if reasoning is not None:
        print('reasoning_len=', len(reasoning))
        print('reasoning_head=', repr(reasoning[:300]))

    # best-effort parse
    text = (ch.message.content or '').strip()
    if not text and reasoning:
        text = reasoning.strip()
    if '```' in text:
        parts = text.split('```')
        for p in parts:
            p = p.strip()
            if p.startswith('json'):
                p = p[4:].strip()
            if p.startswith('{') and p.endswith('}'):
                text = p
                break
    st = text.find('{')
    ed = text.rfind('}')
    if st >= 0 and ed > st:
        text = text[st:ed+1]
    parsed = json.loads(text) if text else {'move': None}
    print('parsed=', parsed)

    mv = parsed.get('move')
    if isinstance(mv, dict):
        r1,c1,r2,c2 = int(mv['r1']), int(mv['c1']), int(mv['r2']), int(mv['c2'])
        if 0 <= r1 <= r2 < R and 0 <= c1 <= c2 < C:
            s = int(np.sum(grid[r1:r2+1, c1:c2+1]))
            area = (r2-r1+1)*(c2-c1+1)
            print('model_move_check=', {'sum':s, 'area':area, 'valid_sum10': s==10})
except Exception as e:
    print('model_error_type=', type(e).__name__)
    print('model_error=', str(e))

print('')
print('=== COMPLETE SOLUTION (LOCAL) ===')
for i, (r1,c1,r2,c2,area,s,avg) in enumerate(cands_sorted, 1):
    print(f'{i:03d}. ({r1},{c1})->({r2},{c2}) area={area} sum={s} conf={avg:.3f}')
