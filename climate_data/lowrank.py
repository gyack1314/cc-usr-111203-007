"""面积加权低秩模态分析（缺测视为未观测，不做填补后当真值）。

目标（训练窗观察位上）：
    min  sum_{(t,j) observed} area_j * (X_tj - mean_j - U_t V_j)^2
    s.t. V^T diag(area) V = I，固定秩 r。

mean_j 只用训练窗内该格已观察值估计；全空列或有效数不足配置的格排除并保留映射。
交替最小二乘（行分数/列载荷轮流加权最小二乘）配合面积加权正交化。
缺测位永不进入目标、不填零。识别不了的情形（观察二部图断开、局部最小二乘秩不足、
整行缺测）显式打标，不用ridge制造看似确定的数值。
"""
from . import linalg as la
from .io import date, number, positive, integer


class AnalysisConfigError(ValueError):
    """资料不满足资料约定时抛出；可识别性问题不在此列，只做标记。"""


def _prepare(v):
    cells = list(v['cells'])
    ids = []
    seen = set()
    for c in cells:
        cid = c['id']
        if not isinstance(cid, str) or not cid or cid in seen:
            raise AnalysisConfigError('格id应非空且唯一')
        seen.add(cid)
        ids.append(cid)
    area_all = [positive(c['area']) for c in cells]
    rows = list(v['rows'])
    days = []
    day_seen = set()
    raw = []
    for r in rows:
        d = date(r['day'])
        if d in day_seen:
            raise AnalysisConfigError('日期重复')
        day_seen.add(d)
        days.append(d)
        vals = r['values']
        if set(vals) != seen:
            raise AnalysisConfigError('行的格引用与cells不齐')
        raw.append([None if vals[cid] is None else number(vals[cid]) for cid in ids])
    windows = {}
    for name, pair in v['windows'].items():
        lo, hi = date(pair[0]), date(pair[1])
        if lo > hi:
            raise AnalysisConfigError('时间窗逆序')
        windows[name] = (lo, hi)
    for required in ('early', 'validation', 'external'):
        if required not in windows:
            raise AnalysisConfigError(f'缺少时间窗: {required}')
    # 三个约定窗互不重叠（闭区间：共享任一日期即视为重叠）。
    pairs = [windows[k] for k in ('early', 'validation', 'external')]
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = pairs[i], pairs[j]
            if not (a[1] < b[0] or b[1] < a[0]):
                raise AnalysisConfigError('early/validation/external必须互不重叠')
    order = sorted(range(len(days)), key=lambda i: days[i])
    days = [days[i] for i in order]
    raw = [raw[i] for i in order]
    ctx = list(v['context_cells'])
    tgt = list(v['target_cells'])
    if not ctx or not tgt:
        raise AnalysisConfigError('context与target均不得为空')
    if set(ctx) & set(tgt) or any(x not in seen for x in ctx + tgt):
        raise AnalysisConfigError('context/target必须不相交且引用现有格')
    rks = [integer(x, minimum=1) for x in v['rank_candidates']]
    if len(set(rks)) != len(rks):
        raise AnalysisConfigError('秩候选不得重复')
    cfg = dict(v['solver'])
    for key in ('tolerance', 'max_iterations', 'starts',
                'rank_tolerance', 'eigen_tie_tolerance'):
        if key not in cfg:
            raise AnalysisConfigError(f'solver缺少配置: {key}')
    if not isinstance(cfg['starts'], list) or not cfg['starts']:
        raise AnalysisConfigError('starts必须是非空列表')
    for s in cfg['starts']:
        integer(s, minimum=1)
    if not (isinstance(cfg['max_iterations'], int) and cfg['max_iterations'] >= 1):
        raise AnalysisConfigError('max_iterations必须是正整数')
    for key in ('tolerance', 'rank_tolerance', 'eigen_tie_tolerance'):
        if not isinstance(cfg[key], (int, float)) or not (cfg[key] > 0):
            raise AnalysisConfigError(f'{key}必须是正数')
    min_obs = integer(v['min_train_observations'], minimum=1)
    return {
        'ids': ids, 'area_all': area_all, 'days': days, 'raw': raw,
        'windows': windows, 'context': ctx, 'target': tgt,
        'ranks': sorted(set(rks)), 'min_obs': min_obs, 'solver': cfg,
    }


def _row_indices(days, lo, hi):
    return [t for t, d in enumerate(days) if lo <= d <= hi]


def _bipartite_components(mask, idx):
    """观察二部图连通分量：边为(t,j)观察，j限定在入选列idx。"""
    T = len(mask)
    pos = {j: k for k, j in enumerate(idx)}
    adj_t = [[] for _ in range(T)]
    adj_j = {j: [] for j in idx}
    for t in range(T):
        for j in idx:
            if mask[t][j]:
                adj_t[t].append(j)
                adj_j[j].append(t)
    seen_t, seen_j, comps = set(), set(), []
    for t0 in range(T):
        if t0 in seen_t or not adj_t[t0]:
            continue
        stack = [('t', t0)]
        ts, js = set(), set()
        while stack:
            kind, k = stack.pop()
            if kind == 't':
                if k in seen_t:
                    continue
                seen_t.add(k); ts.add(k)
                for j in adj_t[k]:
                    if j not in seen_j:
                        stack.append(('j', j))
            else:
                if k in seen_j:
                    continue
                seen_j.add(k); js.add(k)
                for t in adj_j[k]:
                    if t not in seen_t:
                        stack.append(('t', t))
        comps.append((sorted(ts), sorted(js)))
    return comps


def _prepare_window(p, which):
    """切窗、按训练/开发窗自身观察数排除列，返回窗内矩阵与映射。"""
    lo, hi = p['windows'][which]
    tr = _row_indices(p['days'], lo, hi)
    mask_w, Xw = [], []
    for t in tr:
        mask_w.append([p['raw'][t][j] is not None for j in range(len(p['ids']))])
        Xw.append([0.0 if p['raw'][t][j] is None else float(p['raw'][t][j])
                   for j in range(len(p['ids']))])
    keep, dropped = [], []
    for j in range(len(p['ids'])):
        cnt = sum(1 for t in range(len(tr)) if mask_w[t][j])
        if cnt == 0:
            dropped.append({'cell': p['ids'][j], 'observations': cnt, 'reason': '全空列'})
        elif cnt < p['min_obs']:
            dropped.append({'cell': p['ids'][j], 'observations': cnt, 'reason': '有效数不足'})
        else:
            keep.append(j)
    mapping = {p['ids'][j]: k for k, j in enumerate(keep)}
    n = len(keep)
    X = [[Xw[t][j] for j in keep] for t in range(len(tr))]
    mask = [[mask_w[t][j] for j in keep] for t in range(len(tr))]
    area = [p['area_all'][j] for j in keep]
    mean = []
    for k in range(n):
        vals = [X[t][k] for t in range(len(tr)) if mask[t][k]]
        mean.append(sum(vals) / len(vals))
    Y = [[X[t][k] - mean[k] for k in range(n)] for t in range(len(tr))]
    return {
        'name': which, 'row_global': tr, 'days': [p['days'][t] for t in tr],
        'keep': keep, 'mapping': mapping, 'dropped': dropped,
        'X': X, 'mask': mask, 'area': area, 'mean': mean, 'Y': Y,
    }


def _objective(Y, mask, area, U, V):
    """只在观察位求和的加权平方目标；缺测位不参与。"""
    total, nobs = 0.0, 0
    r = len(U[0])
    for t in range(len(Y)):
        Vt = V
        Ut = U[t]
        for k in range(len(area)):
            if not mask[t][k]:
                continue
            pred = sum(Ut[a] * Vt[k][a] for a in range(r))
            e = Y[t][k] - pred
            total += area[k] * e * e
            nobs += 1
    return total, nobs


def _rank_feasibility(mask, r):
    """在拟合前判定秩r是否可识别，绝不靠ridge掩盖结构欠定。

    条件：观察二部图连通；每个含观察的行至少有r个观察列；每列至少有r个观察行；
    且 r 不超过观察行数与列数。整行缺测只使其时间分数不可估，单独标记。
    """
    T, n = len(mask), len(mask[0])
    comps = _bipartite_components(mask, list(range(n)))
    blockers = []
    empty_rows = [t for t in range(T) if not any(mask[t])]
    if len(comps) > 1:
        blockers.append({'kind': 'disconnected_observation_graph',
                         'components': [{'time_index': ts, 'cell_index': js}
                                        for ts, js in comps],
                         'note': '观察二部图不连通，跨分量旋转不可识别，不做整体拟合'})
    short_rows = [t for t in range(T)
                  if 0 < sum(1 for k in range(n) if mask[t][k]) < r]
    if short_rows:
        blockers.append({'kind': 'row_observed_columns_below_rank',
                         'time_index': short_rows,
                         'note': '存在观察列数少于秩r的行，行分数不可识别'})
    short_cols = [k for k in range(n)
                  if sum(1 for t in range(T) if mask[t][k]) < r]
    if short_cols:
        blockers.append({'kind': 'column_observations_below_rank',
                         'cell_index': short_cols,
                         'note': '存在观察行数少于秩r的列，载荷不可识别'})
    active_rows = T - len(empty_rows)
    if r > min(active_rows, n):
        blockers.append({'kind': 'rank_exceeds_dimension',
                         'active_rows': active_rows, 'columns': n, 'requested_rank': r,
                         'note': '秩超过观察行数或列数'})
    return {'feasible': not blockers, 'blockers': blockers,
            'components': comps, 'empty_rows': empty_rows}


def _fit_als(Y, mask, area, r, cfg, seed):
    """固定秩r的多起点之一。返回单起点的局部拟合与完整诊断。"""
    T, n = len(Y), len(area)
    rng = la.SplitMix(seed)
    U = [[rng.gaussian() for _ in range(r)] for _ in range(T)]
    V = [[rng.gaussian() for _ in range(r)] for _ in range(n)]
    V, Binvt, eigvals, ok = la.area_orthonormalize(V, area, cfg['rank_tolerance'])
    U = la.matmul(U, Binvt)
    tol = float(cfg['tolerance'])
    maxit = int(cfg['max_iterations'])
    rank_deficient = []
    prev, _ = _objective(Y, mask, area, U, V)
    initial = prev
    stop_reason = '达到最大迭代'
    iters = 0
    for it in range(1, maxit + 1):
        iters = it
        # 行步：每行在其观察列上加权最小二乘求时间分数。
        for t in range(T):
            obs = [k for k in range(n) if mask[t][k]]
            if not obs:
                U[t] = [0.0] * r
                continue
            A = [[V[k][a] for a in range(r)] for k in obs]
            y = [Y[t][k] for k in obs]
            w = [area[k] for k in obs]
            c, rank, _ = la.weighted_lstsq(A, y, w, cfg['rank_tolerance'])
            U[t] = c
            if rank < r:
                rank_deficient.append({'step': 'row', 'time_index': t, 'rank': rank})
        # 列步：每列在其观察行上求载荷。
        for k in range(n):
            obs = [t for t in range(T) if mask[t][k]]
            A = [[U[t][a] for a in range(r)] for t in obs]
            y = [Y[t][k] for t in obs]
            w = [area[k]] * len(obs)
            c, rank, _ = la.weighted_lstsq(A, y, w, cfg['rank_tolerance'])
            V[k] = c
            if rank < r:
                rank_deficient.append({'step': 'column', 'cell_index': k, 'rank': rank})
        V, Binvt, eigvals, full = la.area_orthonormalize(V, area, cfg['rank_tolerance'])
        if not full:
            rank_deficient.append({'step': 'orthonormalize', 'gram_eigenvalues': eigvals})
        U = la.matmul(U, Binvt)
        cur, nobs = _objective(Y, mask, area, U, V)
        denom = max(1.0, abs(prev))
        if prev - cur < -tol * denom:
            stop_reason = '目标上升（数值异常）'
        if abs(prev - cur) <= tol * denom:
            stop_reason = '相对目标变化低于容差'
        prev = cur
        if stop_reason != '达到最大迭代':
            break
    final, nobs = _objective(Y, mask, area, U, V)
    return {
        'U': U, 'V': V, 'objective': final, 'observations': nobs,
        'initial_objective': initial, 'objective_decreased': final < initial,
        'stop_reason': stop_reason, 'iterations': iters,
        'rank_deficient_events': rank_deficient, 'seed': seed,
        'empty_rows': [t for t in range(T) if not any(mask[t])],
    }


def _identifiability(mask, r, fit):
    """汇总不可识别标识：整行缺测、局部秩不足、观察二部图断开。"""
    T = len(mask)
    n = len(mask[0])
    idx = list(range(n))
    comps = _bipartite_components(mask, idx)
    reasons = []
    if fit['empty_rows']:
        reasons.append({'kind': 'empty_time_row', 'time_index': fit['empty_rows'],
                        'note': '整行缺测，不估时间分数（置零并跳过投影）'})
    deficient = [e for e in fit['rank_deficient_events'] if e['step'] != 'orthonormalize']
    if deficient:
        reasons.append({'kind': 'local_rank_deficiency', 'events': deficient,
                        'note': '行/列局部最小二乘秩低于r，自由分量置零，该量不可识别'})
    if len(comps) > 1:
        reasons.append({'kind': 'disconnected_observation_graph',
                        'components': [{'time_index': ts, 'cell_index': js}
                                       for ts, js in comps],
                        'note': '观察二部图不连通，跨分量模态不可整体识别'})
    # 行内已观察载荷列不足秩。
    short_rows = [t for t in range(T)
                  if 0 < sum(1 for k in range(n) if mask[t][k]) < r]
    if short_rows:
        reasons.append({'kind': 'row_observed_columns_below_rank',
                        'time_index': short_rows, 'note': '该行观察列数少于秩r'})
    return reasons, comps


def _postprocess_modes(fit, area, cell_ids, tie_tol):
    """能量排序、公开符号规则定向、近重根等价子空间分组。"""
    U, V = fit['U'], fit['V']
    r = len(U[0])
    energy = [sum(U[t][a] ** 2 for t in range(len(U))) for a in range(r)]
    perm = sorted(range(r), key=lambda a: (-energy[a], a))
    U = [[U[t][a] for a in perm] for t in range(len(U))]
    V = [[V[k][a] for a in perm] for k in range(len(V))]
    energy = [energy[a] for a in perm]
    # 符号规则：面积加权绝对载荷最大的格取正，同值取最小cell_id。
    signs = []
    for a in range(r):
        best = None
        for k in sorted(range(len(V)), key=lambda k: cell_ids[k]):
            mag = area[k] * abs(V[k][a])
            if best is None or mag > best[0]:
                best = (mag, k)
        s = 1.0 if V[best[1]][a] >= 0.0 else -1.0
        signs.append(s)
        if s < 0.0:
            for k in range(len(V)):
                V[k][a] = -V[k][a]
            for t in range(len(U)):
                U[t][a] = -U[t][a]
    # 近重根：按能量相邻且相对差小于容差分组，组内只声称等价子空间。
    groups = []
    i = 0
    while i < r:
        j = i
        base = energy[i]
        while j + 1 < r and (energy[j] - energy[j + 1]) <= tie_tol * max(1.0, abs(base)):
            j += 1
        groups.append(list(range(i, j + 1)))
        i = j + 1
    projections = []
    for g in groups:
        Vg = [[V[k][a] for a in g] for k in range(len(V))]
        P = la.matmul(Vg, la.transpose(Vg))
        P = [[area[j] * P[i][j] for j in range(len(V))] for i in range(len(V))]
        projections.append({'mode_index': g, 'projection': P})
    return U, V, energy, groups, projections, signs


def _project_row(y_row, ctx_obs, V_rows, area_rows, r, rank_tol):
    """单外行在context格上求分数：min sum area*(y - U_row V)^2。

    target格完全不参与。返回(分数或None, 秩, 状态)。
    """
    if not ctx_obs:
        return None, 0, 'context为空'
    A = [[V_rows[k][a] for a in range(r)] for k in ctx_obs]
    y = [y_row[k] for k in ctx_obs]
    w = [area_rows[k] for k in ctx_obs]
    c, rank, _ = la.weighted_lstsq(A, y, w, rank_tol)
    if rank < r:
        return None, rank, 'context秩不足'
    return c, rank, 'ok'


def _holdout_rows(w_train, w_eval, r, V, mean, p):
    """对评估窗每一行套用context/target规则。

    评估窗按训练窗保留的列对齐，中心化一律使用训练窗均值；评估窗自身的列筛选与均值
    不参与，杜绝任何借未来信息的填补。target格完全不进入该行投影。
    """
    train_keep = w_train['keep']
    pos = {j: k for k, j in enumerate(train_keep)}
    # 评估窗原始观察位，按训练窗列顺序重排。
    emask = [[p['raw'][w_eval['row_global'][t]][j] is not None
              for j in train_keep] for t in range(len(w_eval['days']))]
    eval_raw = [[0.0 if p['raw'][w_eval['row_global'][t]][j] is None
                 else float(p['raw'][w_eval['row_global'][t]][j])
                 for j in train_keep] for t in range(len(w_eval['days']))]
    eval_y = [[eval_raw[t][k] - mean[k] for k in range(len(train_keep))]
              for t in range(len(w_eval['days']))]
    rng_ctx = [pos[j] for j in train_keep
               if p['ids'][j] in set(p['context'])]
    rng_tgt = [pos[j] for j in train_keep
               if p['ids'][j] in set(p['target'])]
    dropped_ctx = [c for c in p['context'] if c not in w_train['mapping']]
    dropped_tgt = [c for c in p['target'] if c not in w_train['mapping']]
    out = []
    for t in range(len(w_eval['days'])):
        day = w_eval['days'][t].isoformat()
        ctx_present = [k for k in rng_ctx if emask[t][k]]
        tgt_present = [k for k in rng_tgt if emask[t][k]]
        entry = {'day': day,
                 'context_observed': [p['ids'][train_keep[k]] for k in ctx_present],
                 'target_observed': [p['ids'][train_keep[k]] for k in tgt_present],
                 'context_dropped_cells': dropped_ctx,
                 'target_dropped_cells': dropped_tgt}
        if not rng_tgt:
            entry.update(status='target格在训练窗被排除，无法评分',
                         score=None, score_row=None)
            out.append(entry)
            continue
        if not tgt_present:
            entry.update(status='target为空', score=None, score_row=None)
            out.append(entry)
            continue
        c, rank, status = _project_row(eval_y[t], ctx_present, V,
                                       w_train['area'], r,
                                       p['solver']['rank_tolerance'])
        if c is None:
            entry.update(status=status, rank=rank, score=None, score_row=None)
            out.append(entry)
            continue
        err = 0.0
        area_sum = 0.0
        details = []
        for k in tgt_present:
            pred = mean[k] + sum(c[a] * V[k][a] for a in range(r))
            e = eval_raw[t][k] - pred
            err += w_train['area'][k] * e * e
            area_sum += w_train['area'][k]
            details.append({'cell': p['ids'][train_keep[k]],
                            'actual': eval_raw[t][k], 'predicted': pred,
                            'area': w_train['area'][k], 'squared_error': e * e})
        entry['status'] = 'ok'
        entry['rank'] = rank
        entry['score'] = err
        entry['score_row'] = err / area_sum
        entry['context_score'] = c
        entry['target_details'] = details
        out.append(entry)
    scored = [e for e in out if e['status'] == 'ok']
    total = sum(e['score'] for e in scored)
    total_area = sum(d['area'] for e in scored for d in e['target_details'])
    return out, scored, total, total_area


def _fit_rank(p, w, r, starts):
    """对给定秩做多起点择优；仅声称局部拟合，不伪称全域最优。"""
    cfg = p['solver']
    fits = [_fit_als(w['Y'], w['mask'], w['area'], r, cfg, seed) for seed in starts]
    best = min(fits, key=lambda f: f['objective'])
    return best, fits


def analyze(v):
    """完整流程：early训练选秩 -> 开发窗重拟合 -> external同规则外期评估。"""
    p = _prepare(v)
    cfg = p['solver']
    starts = [int(s) for s in cfg['starts']]
    tie_tol = float(cfg['eigen_tie_tolerance'])
    early = _prepare_window(p, 'early')
    # 开发窗 = early ∪ validation（含首末日、互不重叠），external不参与。
    # 用独立内部键，避免与资料约定的 validation 单日窗相互覆盖。
    lo = min(p['windows']['early'][0], p['windows']['validation'][0])
    hi = max(p['windows']['early'][1], p['windows']['validation'][1])
    p['windows']['__development__'] = (lo, hi)
    dev = _prepare_window(p, '__development__')
    val = _prepare_window(p, 'validation')
    ext = _prepare_window(p, 'external')

    rank_report = []
    for r in p['ranks']:
        feas = _rank_feasibility(early['mask'], r)
        base = {
            'rank': r,
            'observations': sum(1 for t in range(len(early['mask']))
                                for k in range(len(early['area'])) if early['mask'][t][k]),
            'starts_attempted': 0,
            'local_optimum_only': True,
            'global_optimum_claimed': False,
            'graph_components': len(feas['components']),
            'validation_scored_rows': 0,
            'validation_weighted_error': None,
            'validation_area_normalized_error': None,
            'validation_rows': [],
        }
        if not feas['feasible']:
            base.update(training_objective=None, objective_decreased=None,
                        stop_reason='未拟合：结构不可识别', iterations=0,
                        identifiability=feas['blockers'])
            rank_report.append(base)
            continue
        best, fits = _fit_rank(p, early, r, starts)
        reasons, comps = _identifiability(early['mask'], r, best)
        hold, scored, total, total_area = _holdout_rows(
            early, val, r, best['V'], early['mean'], p)
        base.update(
            training_objective=best['objective'],
            objective_decreased=best['objective_decreased'],
            stop_reason=best['stop_reason'], iterations=best['iterations'],
            starts_attempted=len(fits), identifiability=reasons,
            validation_scored_rows=len(scored),
            validation_weighted_error=total if scored else None,
            validation_area_normalized_error=(total / total_area) if scored else None,
            validation_rows=hold)
        rank_report.append(base)
    selectable = [rep for rep in rank_report if rep['validation_scored_rows'] > 0]
    if not selectable:
        chosen = None
        selection = {'selected_rank': None,
                     'reason': '没有可评分的验证行，不能选秩',
                     'tie_rule': '同分取较小r'}
    else:
        best_err = min(rep['validation_weighted_error'] for rep in selectable)
        tied = [rep for rep in selectable
                if abs(rep['validation_weighted_error'] - best_err) <= tie_tol * max(1.0, abs(best_err))]
        chosen = min(rep['rank'] for rep in tied)
        selection = {'selected_rank': chosen, 'tie_rule': '同分取较小r',
                     'tied_ranks': sorted(rep['rank'] for rep in tied),
                     'scoring_rule': 'context求分数、不重叠target计面积加权误差，target不参与投影'}

    refit = None
    if chosen is not None:
        feas_dev = _rank_feasibility(dev['mask'], chosen)
        if not feas_dev['feasible']:
            refit = {'window': 'development', 'rank': chosen,
                     'fitted': False,
                     'stop_reason': '未重拟合：开发窗结构不可识别',
                     'identifiability': feas_dev['blockers'],
                     'graph_components': len(feas_dev['components']),
                     'local_optimum_only': True}
        else:
            best, fits = _fit_rank(p, dev, chosen, starts)
            reasons, comps = _identifiability(dev['mask'], chosen, best)
            U, V, energy, groups, projections, signs = _postprocess_modes(
                best, dev['area'], [p['ids'][j] for j in dev['keep']], tie_tol)
            refit = {
                'window': 'development', 'fitted': True,
                'rank': chosen,
                'training_objective': best['objective'],
                'observations': best['observations'],
                'objective_decreased': best['objective_decreased'],
                'stop_reason': best['stop_reason'],
                'iterations': best['iterations'],
                'starts_attempted': len(fits),
                'local_optimum_only': True,
                'identifiability': reasons,
                'graph_components': len(comps),
                'modes': [{'index': a, 'energy': energy[a], 'sign': signs[a],
                           'loadings': {p['ids'][dev['keep'][k]]: V[k][a]
                                        for k in range(len(dev['keep']))}}
                          for a in range(chosen)],
                'equivalent_subspaces': [{'mode_index': g} for g in groups if len(g) > 1],
                'projection_matrices': [
                    {'mode_index': proj['mode_index'],
                     'cells': [p['ids'][j] for j in dev['keep']],
                     'matrix': proj['projection']} for proj in projections],
                'scores': {p['days'][dev['row_global'][t]].isoformat():
                           {f'mode_{a}': U[t][a] for a in range(chosen)}
                           for t in range(len(dev['days']))},
                'means': {p['ids'][dev['keep'][k]]: dev['mean'][k]
                          for k in range(len(dev['keep']))},
            }

    external = None
    if refit is not None and refit.get('fitted'):
        V_refit = V
        hold, scored, total, total_area = _holdout_rows(
            dev, ext, chosen, V_refit, dev['mean'], p)
        external = {
            'window': 'external',
            'scoring_rule': '与验证完全相同的context/target规则；external不参与任何拟合',
            'scored_rows': len(scored),
            'weighted_error': total if scored else None,
            'area_normalized_error': (total / total_area) if scored else None,
            'rows': hold,
        }

    out_windows = {}
    for name in ('early', 'validation', 'development', 'external'):
        lo, hi = p['windows'][name if name != 'development' else '__development__']
        out_windows[name] = {
            'start': lo.isoformat(), 'end': hi.isoformat(),
            'rows': [p['days'][t].isoformat()
                     for t in _row_indices(p['days'], lo, hi)]}
    return {
        'source': {'cells': len(p['ids']), 'time_rows': len(p['days'])},
        'windows': out_windows,
        'column_screening': {
            'early': {'kept': [p['ids'][j] for j in early['keep']],
                      'dropped': early['dropped'], 'mapping': early['mapping']},
            'development': {'kept': [p['ids'][j] for j in dev['keep']],
                            'dropped': dev['dropped'], 'mapping': dev['mapping']},
        },
        'mask_archive': _archive_masks(p, early, dev, val, ext),
        'rank_selection': selection,
        'rank_candidates': rank_report,
        'refit': refit,
        'external': external,
    }


def _archive_masks(p, early, dev, val, ext):
    """保存训练/验证/外期日期与观察掩码，并固定context/target格集合。"""
    def pack(w):
        return {'days': [d.isoformat() for d in w['days']],
                'cells': [p['ids'][j] for j in w['keep']],
                'observed': [[bool(w['mask'][t][k]) for k in range(len(w['keep']))]
                             for t in range(len(w['days']))]}
    return {'early': pack(early), 'development': pack(dev),
            'validation': pack(val), 'external': pack(ext),
            'context_cells': p['context'], 'target_cells': p['target']}
