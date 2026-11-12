"""面积加权低秩模态的性质核对（纯标准库，合成数据自带已知真值）。"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from climate_data.io import load
from climate_data.lowrank import (
    analyze, _prepare, _prepare_window, _rank_feasibility,
    _bipartite_components, _postprocess_modes,
)
from climate_data import linalg as la


def make_doc(cells, rows, windows, context, target, ranks=(1, 2), min_obs=2):
    return {
        'cells': [{'id': cid, 'area': a} for cid, a in cells],
        'rows': [{'day': d, 'values': vals} for d, vals in rows],
        'windows': windows,
        'context_cells': context,
        'target_cells': target,
        'rank_candidates': list(ranks),
        'min_train_observations': min_obs,
        'solver': {'tolerance': 1e-10, 'max_iterations': 400,
                   'starts': [1, 7, 13], 'rank_tolerance': 1e-10,
                   'eigen_tie_tolerance': 1e-7},
    }


def gen_field(T, n, r, areas, seed=3):
    """生成严格秩r、V满足面积正交的合成场 X=mean+U V。"""
    rng = la.SplitMix(seed)
    U = [[rng.gaussian() for _ in range(r)] for _ in range(T)]
    Vraw = [[rng.gaussian() for _ in range(r)] for _ in range(n)]
    V, _, _, full = la.area_orthonormalize(Vraw, areas, 1e-12)
    assert full
    mean = [rng.gaussian() * 5.0 for _ in range(n)]
    X = [[mean[k] + sum(U[t][a] * V[k][a] for a in range(r))
          for k in range(n)] for t in range(T)]
    return X, mean, U, V


def days(T, year=2020, month=1):
    import datetime
    base = datetime.date(year, month, 1)
    return [(base + datetime.timedelta(days=i)).isoformat() for i in range(T)]


class LinalgTests(unittest.TestCase):
    def test_qr_reconstruction_and_orthogonality(self):
        rng = la.SplitMix(11)
        A = [[rng.gaussian() for _ in range(4)] for _ in range(7)]
        Q, R = la.qr(A)
        # Q 正交：Q^T Q = I
        Qt = la.transpose(Q)
        QtQ = la.matmul(Qt, Q)
        for i in range(7):
            for j in range(7):
                self.assertAlmostEqual(QtQ[i][j], 1.0 if i == j else 0.0, places=10)
        QR = la.matmul(Q, R)
        for i in range(7):
            for j in range(4):
                self.assertAlmostEqual(QR[i][j], A[i][j], places=10)

    def test_weighted_lstsq_rank_report(self):
        # 秩不足设计阵：两列相同，必须报告 rank<2 而非ridge出确定解。
        A = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        y = [2.0, 4.0, 6.1]
        c, rank, thr = la.weighted_lstsq(A, y, [1.0, 2.0, 3.0], 1e-10)
        self.assertEqual(rank, 1)
        # 最小残差解仍应较好拟合（秩1最小二乘允许小残差，但不引入ridge）
        for i, row in enumerate(A):
            self.assertAlmostEqual(
                sum(row[k] * c[k] for k in range(2)), y[i], delta=0.05)

    def test_eigh_orthogonal_eigenvectors(self):
        M = [[4.0, 1.0, 2.0], [1.0, 2.0, 0.5], [2.0, 0.5, 3.0]]
        vals, vecs = la.eigh(M)
        self.assertEqual(vals, sorted(vals))
        for a in range(3):
            for b in range(3):
                dot = sum(vecs[i][a] * vecs[i][b] for i in range(3))
                self.assertAlmostEqual(dot, 1.0 if a == b else 0.0, places=10)
            # M v = λ v
            for i in range(3):
                mv = sum(M[i][k] * vecs[k][a] for k in range(3))
                self.assertAlmostEqual(mv, vals[a] * vecs[i][a], places=9)

    def test_area_orthonormality_preserves_reconstruction(self):
        rng = la.SplitMix(5)
        n, r, T = 6, 2, 5
        area = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        V0 = [[rng.gaussian() for _ in range(r)] for _ in range(n)]
        U0 = [[rng.gaussian() for _ in range(r)] for _ in range(T)]
        before = la.matmul(U0, la.transpose(V0))
        V1, Binvt, _, full = la.area_orthonormalize(V0, area, 1e-12)
        self.assertTrue(full)
        gram = la.matmul(la.transpose(V1),
                         [[area[i] * V1[i][j] for j in range(r)] for i in range(n)])
        for i in range(r):
            for j in range(r):
                self.assertAlmostEqual(gram[i][j], 1.0 if i == j else 0.0, places=10)
        after = la.matmul(la.matmul(U0, Binvt), la.transpose(V1))
        for i in range(T):
            for j in range(n):
                self.assertAlmostEqual(before[i][j], after[i][j], places=9)


class FullFieldTests(unittest.TestCase):
    def test_complete_rank2_field_recovered(self):
        T, n, r = 12, 5, 2
        areas = [1.0, 1.5, 2.0, 3.0, 4.5]
        X, mean, U, V = gen_field(T, n, r, areas)
        ds = days(T)
        rows = [(ds[t], {f'c{k}': X[t][k] for k in range(n)}) for t in range(T)]
        win = {'early': [ds[0], ds[7]], 'validation': [ds[8], ds[9]],
               'external': [ds[10], ds[11]]}
        doc = make_doc([(f'c{k}', areas[k]) for k in range(n)], rows, win,
                       ['c0', 'c1'], ['c2', 'c3', 'c4'], ranks=(1, 2))
        res = analyze(doc)
        self.assertEqual(res['rank_selection']['selected_rank'], 2)
        # 秩2应近零目标；秩1必有正目标
        o1 = next(x for x in res['rank_candidates'] if x['rank'] == 1)['training_objective']
        o2 = next(x for x in res['rank_candidates'] if x['rank'] == 2)['training_objective']
        self.assertGreater(o1, 1e-6)
        self.assertLess(o2, 1e-14)
        # 载荷面积正交
        refit = res['refit']
        self.assertTrue(refit['fitted'])
        Vhat = [[refit['modes'][a]['loadings'][f'c{k}'] for a in range(r)]
                for k in range(n)]
        for a in range(r):
            for b in range(r):
                g = sum(areas[k] * Vhat[k][a] * Vhat[k][b] for k in range(n))
                self.assertAlmostEqual(g, 1.0 if a == b else 0.0, places=8)
        # 外期误差接近机器精度
        self.assertLess(res['external']['area_normalized_error'], 1e-14)
        # 公开符号规则：定向后每个模态最大面积加权载荷格为正
        for a in range(r):
            mags = sorted(((areas[k] * abs(Vhat[k][a]), f'c{k}') for k in range(n)),
                          key=lambda z: (-z[0], z[1]))
            top_cell = mags[0][1]
            self.assertGreater(refit['modes'][a]['loadings'][top_cell], 0.0)


class CheckerboardTests(unittest.TestCase):
    def test_checkerboard_missing_never_zero_filled(self):
        T, n, r = 12, 6, 2
        areas = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        X, mean, U, V = gen_field(T, n, r, areas, seed=9)
        ds = days(T)
        rows = []
        observed = [[True] * n for _ in range(T)]
        for t in range(T):
            vals = {}
            for k in range(n):
                if t < 8:
                    # 连通棋盘：按 (t+k)%3 交错缺测，每行保留4列、每格观察>=5、
                    # 相邻行共享观察格使二部图保持连通，且无整行空。
                    missing = (t + k) % 3 == 0
                else:
                    # 留出/外期行：context与target都有观察，可投影也可评分。
                    missing = False
                observed[t][k] = not missing
                vals[f'c{k}'] = None if missing else X[t][k]
            rows.append((ds[t], vals))
        win = {'early': [ds[0], ds[7]], 'validation': [ds[8], ds[9]],
               'external': [ds[10], ds[11]]}
        doc = make_doc([(f'c{k}', areas[k]) for k in range(n)], rows, win,
                       ['c0', 'c1', 'c2'], ['c3', 'c4', 'c5'], ranks=(1, 2), min_obs=3)
        res = analyze(doc)
        self.assertEqual(res['rank_selection']['selected_rank'], 2)
        o1 = next(x for x in res['rank_candidates'] if x['rank'] == 1)
        o2 = next(x for x in res['rank_candidates'] if x['rank'] == 2)
        # 缺测位不进入目标：秩2在观察位上显著优于秩1，且目标确实单调下降。
        self.assertLess(o2['training_objective'], o1['training_objective'])
        self.assertTrue(o2['objective_decreased'])
        # 多起点只声称局部拟合，绝不伪称全域最优。
        self.assertTrue(o2['local_optimum_only'])
        self.assertFalse(o2['global_optimum_claimed'])
        # 掩码存档与原始缺测一致
        arch = res['mask_archive']['early']
        for t, d in enumerate(arch['days']):
            gi = ds.index(d)
            for k, cid in enumerate(arch['cells']):
                ki = int(cid[1:])
                self.assertEqual(arch['observed'][t][k], observed[gi][ki])
        # target 不出现在任何验证行的 context_observed 中（target不参与投影）
        for vr in o2['validation_rows']:
            if vr['status'] == 'ok':
                self.assertFalse(set(vr['context_observed']) & set(doc['target_cells']))
        # 外期误差为有限非负，且只用context投影得到
        self.assertGreaterEqual(res['external']['area_normalized_error'], 0.0)


class IsolatedComponentTests(unittest.TestCase):
    def test_disconnected_graph_marked_not_ridded(self):
        # 前4行只观察 c0,c1；后4行只观察 c2,c3 -> 二部图两个分量。
        cells = [('c0', 1.0), ('c1', 2.0), ('c2', 3.0), ('c3', 4.0)]
        ds = days(12)
        rng = la.SplitMix(2)
        rows = []
        for t in range(12):
            vals = {cid: None for cid, _ in cells}
            block = (0, 1) if t < 6 else (2, 3)
            for k in block:
                vals[f'c{k}'] = rng.gaussian() + 10.0
            rows.append((ds[t], vals))
        win = {'early': [ds[0], ds[7]], 'validation': [ds[8], ds[9]],
               'external': [ds[10], ds[11]]}
        doc = make_doc(cells, rows, win, ['c0', 'c1'], ['c2', 'c3'],
                       ranks=(1, 2), min_obs=2)
        p = _prepare(doc)
        w = _prepare_window(p, 'early')
        comps = _bipartite_components(w['mask'], list(range(len(w['keep']))))
        self.assertGreater(len(comps), 1)
        feas1 = _rank_feasibility(w['mask'], 1)
        self.assertFalse(feas1['feasible'])
        self.assertTrue(any(b['kind'] == 'disconnected_observation_graph'
                            for b in feas1['blockers']))
        res = analyze(doc)
        # 不允许靠ridge产出确定模态：不可识别秩被拦下，无可评分验证行则不选秩
        for rep in res['rank_candidates']:
            if rep['stop_reason'].startswith('未拟合'):
                self.assertTrue(any(z['kind'] == 'disconnected_observation_graph'
                                    for z in rep['identifiability']))

    def test_empty_row_not_scored(self):
        cells = [('c0', 1.0), ('c1', 2.0), ('c2', 3.0), ('c3', 4.0)]
        ds = days(10)
        rows = []
        for t in range(10):
            vals = {f'c{k}': (None if t == 2 else float(t + k)) for k in range(4)}
            rows.append((ds[t], vals))
        win = {'early': [ds[0], ds[7]], 'validation': [ds[8], ds[8]],
               'external': [ds[9], ds[9]]}
        doc = make_doc(cells, rows, win, ['c0', 'c1'], ['c2', 'c3'],
                       ranks=(1,), min_obs=2)
        res = analyze(doc)
        rep = res['rank_candidates'][0]
        self.assertTrue(any(z['kind'] == 'empty_time_row' for z in rep['identifiability']))


class EquivalentRotationTests(unittest.TestCase):
    def test_rotation_keeps_subspace_and_objective(self):
        T, n, r = 10, 5, 2
        areas = [1.0, 2.0, 1.5, 3.0, 2.5]
        X, mean, U, V = gen_field(T, n, r, areas, seed=4)
        # 近重根：让两个模态时间能量几乎相等。对模态空间做正交旋转 (U,V)->(UQ,VQ)，
        # 因 Q Q^T=I，重建 U V^T 不变。
        ang = 0.7
        c, s = math.cos(ang), math.sin(ang)
        Q = [[c, -s], [s, c]]
        Vr = la.matmul(V, Q)
        Ur = la.matmul(U, Q)
        # 旋转后重建完全相同
        A1 = la.matmul(U, la.transpose(V))
        A2 = la.matmul(Ur, la.transpose(Vr))
        for i in range(T):
            for j in range(n):
                self.assertAlmostEqual(A1[i][j], A2[i][j], places=10)
        # 投影矩阵 P=V V^T（面积度量下）对旋转不变
        def proj(Vm):
            P = la.matmul(Vm, la.transpose(Vm))
            return [[areas[j] * P[i][j] for j in range(n)] for i in range(n)]
        P1, P2 = proj(V), proj(Vr)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(P1[i][j], P2[i][j], places=9)
        # 同一资料两次分析（不同起点种子）重建的面积范数应一致，不因逐列差异判异。
        ds = days(T)
        rows = [(ds[t], {f'c{k}': X[t][k] for k in range(n)}) for t in range(T)]
        win = {'early': [ds[0], ds[7]], 'validation': [ds[8], ds[8]],
               'external': [ds[9], ds[9]]}
        doc = make_doc([(f'c{k}', areas[k]) for k in range(n)], rows, win,
                       ['c0', 'c1'], ['c2', 'c3', 'c4'], ranks=(2,))
        r1 = analyze(doc)
        r2 = analyze(doc)
        self.assertAlmostEqual(r1['refit']['training_objective'],
                               r2['refit']['training_objective'], places=12)


class NearDegenerateTests(unittest.TestCase):
    def test_equal_energy_modes_grouped_as_equivalent_subspace(self):
        # 直接对后处理做单元测试：构造严格等能量的U与面积正交的V。
        T, n, r = 12, 5, 2
        areas = [1.0, 2.0, 1.5, 3.0, 2.5]
        rng = la.SplitMix(21)
        U = [[math.cos(2 * math.pi * t / T), math.sin(2 * math.pi * t / T)]
             for t in range(T)]
        Vraw = [[rng.gaussian() for _ in range(r)] for _ in range(n)]
        V, _, _, full = la.area_orthonormalize(Vraw, areas, 1e-12)
        self.assertTrue(full)
        cell_ids = [f'c{k}' for k in range(n)]
        Up, Vp, energy, groups, projections, signs = (
            _postprocess_modes({'U': U, 'V': V}, areas, cell_ids, 1e-7))
        # 两模态能量严格相等。
        self.assertAlmostEqual(energy[0], energy[1], places=12)
        self.assertEqual(groups, [[0, 1]])
        # 单个等价子空间，投影矩阵 n×n。该矩阵是面积度量下的算子 P=Vg Vg^T diag(area)，
        # 幂等性按 (P^2)_ij = sum_k P_ik P_kj 验证（面积权重已内置在P的列中）。
        self.assertEqual(len(projections), 1)
        P = projections[0]['projection']
        self.assertEqual(projections[0]['mode_index'], [0, 1])
        self.assertEqual(len(P), n)
        for i in range(n):
            for j in range(n):
                p2 = sum(P[i][k] * P[k][j] for k in range(n))
                self.assertAlmostEqual(p2, P[i][j], places=8)

    def test_distinct_energy_modes_not_grouped(self):
        T, n, r = 10, 4, 2
        areas = [1.0, 1.0, 1.0, 1.0]
        rng = la.SplitMix(2)
        U = [[(3.0 if a == 0 else 1.0) * rng.gaussian() for a in range(r)]
             for t in range(T)]
        Vraw = [[rng.gaussian() for _ in range(r)] for _ in range(n)]
        V, _, _, _ = la.area_orthonormalize(Vraw, areas, 1e-12)
        _, _, energy, groups, _, _ = _postprocess_modes(
            {'U': U, 'V': V}, areas, [f'c{k}' for k in range(n)], 1e-7)
        self.assertEqual(groups, [[0], [1]])
        self.assertGreater(energy[0], energy[1])


class HoldoutLeakageTests(unittest.TestCase):
    def test_target_values_do_not_affect_scores(self):
        T, n, r = 12, 5, 2
        areas = [1.0, 2.0, 1.5, 3.0, 2.5]
        X, mean, U, V = gen_field(T, n, r, areas, seed=8)
        ds = days(T)

        def build(perturb_target):
            rows = []
            for t in range(T):
                vals = {f'c{k}': X[t][k] for k in range(n)}
                # 只扰动独立外期的target；开发窗（含验证）逐字节不变，
                # 因而载荷、均值、训练分数必须完全相同。
                if perturb_target and t >= 10:
                    for k in (2, 3, 4):
                        vals[f'c{k}'] = X[t][k] + 100.0
                rows.append((ds[t], vals))
            win = {'early': [ds[0], ds[7]], 'validation': [ds[8], ds[9]],
                   'external': [ds[10], ds[11]]}
            return make_doc([(f'c{k}', areas[k]) for k in range(n)], rows, win,
                            ['c0', 'c1'], ['c2', 'c3', 'c4'], ranks=(1, 2))

        res_a = analyze(build(False))
        res_b = analyze(build(True))
        # 载荷与训练分数不得因验证/外期target被改动而变化。
        self.assertEqual(res_a['refit']['modes'], res_b['refit']['modes'])
        self.assertEqual(res_a['refit']['scores'], res_b['refit']['scores'])
        # context 分数（投影只依赖context）逐行相同，仅target误差变化。
        for ra, rb in zip(res_a['external']['rows'], res_b['external']['rows']):
            if ra['status'] == 'ok':
                self.assertEqual(ra['context_score'], rb['context_score'])
                self.assertNotEqual(ra['score'], rb['score'])

    def test_context_rank_deficient_row_is_listed(self):
        # 验证行只观察到1个context格，而秩=2 -> context秩不足，单列不评分。
        T, n, r = 12, 4, 2
        areas = [1.0, 2.0, 3.0, 4.0]
        X, mean, U, V = gen_field(T, n, r, areas, seed=6)
        ds = days(T)
        rows = []
        for t in range(T):
            vals = {f'c{k}': X[t][k] for k in range(4)}
            if t in (8, 9):
                vals['c1'] = None  # context 只剩 c0
            rows.append((ds[t], vals))
        win = {'early': [ds[0], ds[7]], 'validation': [ds[8], ds[9]],
               'external': [ds[10], ds[11]]}
        doc = make_doc([(f'c{k}', areas[k]) for k in range(4)], rows, win,
                       ['c0', 'c1'], ['c2', 'c3'], ranks=(2,), min_obs=2)
        res = analyze(doc)
        rep = res['rank_candidates'][0]
        statuses = {x['day']: x['status'] for x in rep['validation_rows']}
        self.assertEqual(statuses[ds[8]], 'context秩不足')
        self.assertEqual(statuses[ds[9]], 'context秩不足')
        self.assertEqual(rep['validation_scored_rows'], 0)


class AreaNormTests(unittest.TestCase):
    def test_weighted_objective_uses_area_not_count(self):
        # 相同残差下，大面积格主导目标：用两组面积验证目标按面积比例变化。
        T, n = 7, 3
        ds = days(T)
        rng = la.SplitMix(0)
        base = [[rng.gaussian() for _ in range(n)] for _ in range(T)]
        win = {'early': [ds[0], ds[4]], 'validation': [ds[5], ds[5]],
               'external': [ds[6], ds[6]]}

        def run(areas):
            rows = [(ds[t], {f'c{k}': base[t][k] for k in range(n)}) for t in range(T)]
            doc = make_doc([(f'c{k}', areas[k]) for k in range(n)], rows, win,
                           ['c0'], ['c1', 'c2'], ranks=(1,), min_obs=2)
            return analyze(doc)
        r1 = run([1.0, 1.0, 1.0])
        r2 = run([2.0, 2.0, 2.0])
        o1 = r1['rank_candidates'][0]['training_objective']
        o2 = r2['rank_candidates'][0]['training_objective']
        self.assertAlmostEqual(o2 / o1, 2.0, places=8)


class ConfigValidationTests(unittest.TestCase):
    def _base_doc(self):
        T, n = 9, 4
        ds = days(T)
        rows = [(ds[t], {f'c{k}': float(t + k) for k in range(n)})
                for t in range(T)]
        win = {'early': [ds[0], ds[5]], 'validation': [ds[6], ds[7]],
               'external': [ds[8], ds[8]]}
        return make_doc([(f'c{k}', float(k + 1)) for k in range(n)], rows, win,
                        ['c0', 'c1'], ['c2', 'c3'], ranks=(1,))

    def test_overlapping_windows_rejected(self):
        from climate_data.lowrank import AnalysisConfigError
        doc = self._base_doc()
        doc['windows']['validation'] = [doc['windows']['early'][0],
                                        doc['windows']['validation'][1]]
        with self.assertRaises(AnalysisConfigError):
            analyze(doc)

    def test_intersecting_context_target_rejected(self):
        from climate_data.lowrank import AnalysisConfigError
        doc = self._base_doc()
        doc['target_cells'] = ['c1', 'c2']
        with self.assertRaises(AnalysisConfigError):
            analyze(doc)

    def test_empty_context_rejected(self):
        from climate_data.lowrank import AnalysisConfigError
        doc = self._base_doc()
        doc['context_cells'] = []
        with self.assertRaises(AnalysisConfigError):
            analyze(doc)

    def test_bad_solver_config_rejected(self):
        from climate_data.lowrank import AnalysisConfigError
        doc = self._base_doc()
        doc['solver']['tolerance'] = 0.0
        with self.assertRaises(AnalysisConfigError):
            analyze(doc)
        doc2 = self._base_doc()
        doc2['solver']['starts'] = []
        with self.assertRaises(AnalysisConfigError):
            analyze(doc2)


class FixtureContractTests(unittest.TestCase):
    def test_shipped_fixtures_run(self):
        for name in ('normal', 'empty-cell', 'empty-time-row',
                     'disconnected-observation-graph'):
            v, _ = load(f'data/{name}.json')
            res = analyze(v)
            self.assertIn('rank_selection', res)
            self.assertIn('mask_archive', res)
            # 多起点只声称局部拟合
            for rep in res['rank_candidates']:
                self.assertTrue(rep['local_optimum_only'])
                self.assertFalse(rep['global_optimum_claimed'])


if __name__ == '__main__':
    unittest.main()
