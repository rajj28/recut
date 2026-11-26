"""recut: find which hook is winning in a competitor's ads, using only what an ad library shows
(the creative, its start date, and whether it is still running).

Two things break "sort by days running":
  1. Recuts. A winning ad comes back with a new 3-second hook, a new aspect ratio, a zoom, a mirror
     and new burned-in captions. Frame hashes see unrelated creatives.
  2. Relaunches. Media buyers duplicate a winner into a new ad set and pause the original, so the
     winner's ad IDs keep getting reset, and recent winners are still running (right-censored).

Pipeline:
  decode -> content box (strip black / blur bars) -> caption-robust shot cuts -> keyframe SIFT
  -> one ANN index over all descriptors -> RANSAC similarity check that tolerates crop, reframe,
     zoom and mirror and rejects caption-only matches -> shot clusters
  -> drop brand boilerplate (shared end cards) -> link ads that share most of their body
  -> variant families -> hook = first shot -> censoring-aware hook hazard (Gamma-Poisson).

CLI on real data:  python recut.py ads.csv --obs 2026-11-30 --out report.html
  ads.csv columns: ad_id,file,start,end   (dates YYYY-MM-DD, end empty = still running)
"""
from __future__ import annotations

import base64, csv, datetime as dt, html, json, subprocess, sys, time
from collections import Counter, defaultdict

import cv2
import numpy as np

FPS, LONG_SIDE, KF_H, TL_H = 10, 320, 200, 48
# shot matching: SIFT correspondences from one global ANN index, geometric verification, then a
# temporal same-clip check. yspan and tau_t are each the 2.5th percentile of true matches on dev.
M = dict(k=16, dist=200, min_corr=8, inliers=12, scale=(0.7, 1.45), max_rot=6.0, yspan=0.15,
         motion=1.0, tau_t=0.2)
P = dict(boiler=0.25, lo=0.3, hi=0.5)   # family linking; all thresholds tuned on the dev set only
_SIFT = []

# ---------------------------------------------------------------- decode / fingerprint
def probe(path):
    o = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=width,height', '-of', 'csv=p=0', str(path)],
                       capture_output=True, text=True).stdout.strip().split(',')
    return int(o[0]), int(o[1])


def decode(path, fps=FPS, long_side=LONG_SIDE):
    w0, h0 = probe(path)
    k = long_side / max(w0, h0)
    w, h = max(2, round(w0 * k / 2) * 2), max(2, round(h0 * k / 2) * 2)
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'fps={fps},scale={w}:{h}',
                          '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def _longest_run(on):
    best, s = (0, len(on)), None
    bl = 0
    for i, v in enumerate(np.append(on, False)):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s > bl:
                bl, best = i - s, (s, i)
            s = None
    return best


def content_box(fr):
    """Where the real picture is. Black bars and blurred 'TikTok' bars carry almost no
    high-frequency energy, and padding bars come in symmetric pairs. Captions sitting in a bar form
    a separate run and are dropped. A dark sky or floor (one-sided) is not mistaken for a bar."""
    sub = fr[::max(1, len(fr) // 40)]
    lap = np.mean([np.abs(cv2.Laplacian(f, cv2.CV_32F)) for f in sub], axis=0)
    out = []
    for prof in (lap.mean(1), lap.mean(0)):
        n = len(prof)
        a, b = _longest_run(prof > 0.3 * np.percentile(prof, 90))
        if not (a >= 0.05 * n and n - b >= 0.05 * n and abs(a - (n - b)) <= 0.1 * n):
            a, b = 0, n
        out += [a, b]
    return tuple(int(v) for v in out)            # y0, y1, x0, x1


def detect_shots(fr, box, min_len=3):
    y0, y1, x0, x1 = box
    c = fr[:, y0:y1, x0:x1]
    sm = np.stack([cv2.resize(f, (16, 16), interpolation=cv2.INTER_AREA) for f in c]).astype(np.float32)
    hs = np.stack([np.histogram(cv2.resize(f, (48, 48)), 32, (0, 256))[0] for f in c]).astype(np.float32)
    hs /= hs.sum(1, keepdims=True)
    if len(c) < 2:
        return [(0, len(c))]
    # a cut changes most of the frame; a caption swap or a sticker changes a few tiles, so take a
    # high-but-not-max quantile over 16 tiles instead of the frame mean
    td = np.abs(np.diff(sm, axis=0)).reshape(-1, 4, 4, 4, 4).mean((2, 4)).reshape(len(sm) - 1, 16)
    d1 = np.quantile(td, 0.4, axis=1) / 255.0
    d2 = 0.5 * np.abs(np.diff(hs, axis=0)).sum(1)
    med = float(np.median(d1))
    cut = ((d1 > max(0.07, 3 * med)) & (d2 > 0.12)) | (d1 > max(0.18, 5 * med))
    b = [0]
    for i in np.flatnonzero(cut):
        if i + 1 - b[-1] >= min_len:
            b.append(int(i + 1))
    if len(c) - b[-1] < min_len and len(b) > 1:
        b.pop()
    b.append(len(c))
    return list(zip(b[:-1], b[1:]))


def _features(crop):
    """SIFT on the content crop scaled to a fixed height (plus its mirror image). 9:16, 1:1 and 4:5
    crops keep the full source height, so after this only the zoom changes scale between recuts."""
    if not _SIFT:
        _SIFT.append(cv2.SIFT_create(nfeatures=400, contrastThreshold=0.02))
    im = cv2.resize(crop, (max(8, round(crop.shape[1] * KF_H / crop.shape[0])), KF_H), interpolation=cv2.INTER_AREA)
    out = {'shape': im.shape}
    for tag, img in (('', im), ('m', np.ascontiguousarray(im[:, ::-1]))):
        kp, d = _SIFT[0].detectAndCompute(img, None)
        out['p' + tag] = np.float32([q.pt for q in kp]).reshape(-1, 2)
        out['d' + tag] = np.clip(np.round(d if d is not None else np.zeros((0, 128))), 0, 255).astype(np.uint8)
    return out


def fingerprint(path):
    fr = decode(path)
    box = content_box(fr)
    y0, y1, x0, x1 = box
    shots = []
    for s, e in detect_shots(fr, box):
        n = e - s
        k = 3 if n >= 6 else (2 if n >= 3 else 1)
        idx = sorted({s + int((j + 1) * n / (k + 1)) for j in range(k)})
        w = max(8, round((x1 - x0) * TL_H / max(1, y1 - y0)))
        tl = np.stack([cv2.resize(fr[i, y0:y1, x0:x1], (w, TL_H), interpolation=cv2.INTER_AREA) for i in range(s, e)])
        shots.append(dict(t0=s / FPS, t1=e / FPS, kf=[_features(fr[i, y0:y1, x0:x1]) for i in idx], tl=tl))
    return dict(path=str(path), dur=len(fr) / FPS, box=box, ar=(x1 - x0) / max(1, y1 - y0), shots=shots)


# ---------------------------------------------------------------- shot matching
def _verify(pa, pb, shape, m=M):
    """RANSAC similarity transform. Returns (inliers, rows, cols, vertical span) where rows/cols are
    how many rows/columns of a 3x3 grid the inliers touch and the span is their 10-90th percentile
    height as a fraction of the frame. 0 inliers if the transform is not something a recut can do
    (rotation, scale outside the zoom range)."""
    T, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if T is None:
        return 0, 0, 0, 0.0, None
    sc, rot = np.hypot(T[0, 0], T[1, 0]), abs(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    if not (m['scale'][0] < sc < m['scale'][1] and rot < m['max_rot']):
        return 0, 0, 0, 0.0, None
    I = pa[inl.ravel() == 1]
    if len(I) < 2:
        return 0, 0, 0, 0.0, None
    gy = np.clip((I[:, 1] * 3 / shape[0]).astype(int), 0, 2)
    gx = np.clip((I[:, 0] * 3 / shape[1]).astype(int), 0, 2)
    span = float(np.subtract(*np.percentile(I[:, 1], [90, 10]))) / shape[0]
    return len(I), len(set(gy.tolist())), len(set(gx.tolist())), span, T


def _accept(v, m=M):
    """Burned-in captions and subtitles repeat across unrelated ads in the same font and place.
    They sit in one or two text lines, so a real footage match must cover >= 2 rows and 2 columns
    of the frame and spread vertically beyond a text band."""
    n, rows, cols, span = v[:4]
    return n >= m['inliers'] and rows >= 2 and cols >= 2 and span >= m['yspan']


def _temporal(A, B, T, mir, m=M):
    """Same clip, or just the same set? A spatial match proves two shots show the same place; a UGC
    creator filming ten scripts against one wall passes that too. Warp shot B onto shot A with the
    verified transform and correlate their frame-difference images (where things move) at the best
    time offset. The same clip moves the same way; another take does not. Returns the median
    correlation, or None if neither shot moves enough to tell (then the spatial match stands)."""
    a, b = A['tl'].astype(np.float32), B['tl'].astype(np.float32)
    if len(a) < 4 or len(b) < 4:
        return None
    if mir:
        b = b[:, :, ::-1]
    Ts = T.astype(np.float32).copy()
    Ts[:, 2] *= TL_H / KF_H
    h, w = a.shape[1:]
    fl = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    wb = np.stack([cv2.warpAffine(np.ascontiguousarray(f), Ts, (w, h), flags=fl) for f in b])
    valid = cv2.warpAffine(np.ones(b.shape[1:], np.float32), Ts, (w, h), flags=fl) > 0.99
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if valid.sum() < 0.2 * h * w:
        return None
    da, db = np.diff(a, axis=0)[:, valid], np.diff(wb, axis=0)[:, valid]
    if max(np.median(np.abs(da).mean(1)), np.median(np.abs(db).mean(1))) < m['motion']:
        return None
    unit = lambda X: (X - X.mean(1, keepdims=True)) / (np.linalg.norm(X - X.mean(1, keepdims=True), axis=1, keepdims=True) + 1e-6)
    C = unit(da) @ unit(db).T
    return max(float(np.median(np.diagonal(C, L))) for L in range(-(C.shape[0] - 3), C.shape[1] - 2))


def match_shots(fps_, m=M, detail=None):
    """All-vs-all shot matching without all-vs-all comparisons. Every keyframe descriptor and its
    mirror go into one approximate-NN index; each plain descriptor's neighbours in other files are
    putative correspondences, grouped by keyframe pair (and mirror flag), and each group with enough
    of them is verified geometrically. Returns ({(shot_a, shot_b): inliers}, groups verified)."""
    shot_file, kf_shot, shapes, D, Pt, KF, MIR = [], [], [], [], [], [], []
    for fi, f in enumerate(fps_):
        for s in f['shots']:
            si = len(shot_file); shot_file.append(fi)
            for kf in s['kf']:
                ki = len(kf_shot); kf_shot.append(si); shapes.append(kf['shape'])
                for mir, tag in ((0, ''), (1, 'm')):
                    D.append(kf['d' + tag]); Pt.append(kf['p' + tag])
                    KF.append(np.full(len(kf['d' + tag]), ki)); MIR.append(np.full(len(kf['d' + tag]), mir))
    D, Pt = np.vstack(D).astype(np.float32), np.vstack(Pt)
    KF, MIR = np.concatenate(KF).astype(np.int64), np.concatenate(MIR).astype(np.int64)
    shot_file, kf_shot, nk = np.array(shot_file), np.array(kf_shot), len(kf_shot)
    index = cv2.flann_Index(D, dict(algorithm=1, trees=4))
    q = np.flatnonzero(MIR == 0)
    nn, d2 = index.knnSearch(D[q], m['k'], params=dict(checks=48))
    qq, nn = np.repeat(q, m['k']), nn.ravel().astype(np.int64)
    dd = np.sqrt(np.maximum(d2.ravel(), 0))                 # FLANN reports squared L2
    ok = (dd < m['dist']) & (shot_file[kf_shot[KF[qq]]] != shot_file[kf_shot[KF[nn]]])
    qq, nn, dd = qq[ok], nn[ok], dd[ok]
    o = np.argsort(dd, kind='stable'); qq, nn = qq[o], nn[o]
    grp = KF[qq] * (2 * nk) + KF[nn] * 2 + MIR[nn]
    for use_q in (True, False):                             # one-to-one: nearest wins on both sides
        _, first = np.unique(grp * len(D) + (qq if use_q else nn), return_index=True)
        keep = np.sort(first); qq, nn, grp = qq[keep], nn[keep], grp[keep]
    o = np.argsort(grp, kind='stable'); qq, nn, grp = qq[o], nn[o], grp[o]
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]]); ends = np.r_[starts[1:], len(grp)]
    score, n_ver = {}, 0
    for a0, a1 in zip(starts, ends):
        if a1 - a0 < m['min_corr']:
            continue
        n_ver += 1
        ka, kb = KF[qq[a0]], KF[nn[a0]]
        v = _verify(Pt[qq[a0:a1]], Pt[nn[a0:a1]], shapes[ka], m)
        key = tuple(sorted((int(kf_shot[ka]), int(kf_shot[kb]))))
        if detail is not None:
            detail.setdefault(key, []).append(v[:4])
        if _accept(v, m) and v[0] > score.get(key, (0,))[0]:
            score[key] = (v[0], int(kf_shot[ka]), int(kf_shot[kb]), v[4], int(MIR[nn[a0]]))
    shots = [s for f in fps_ for s in f['shots']]
    out = {}
    for key, (n, sa, sb, T, mir) in score.items():
        t = _temporal(shots[sa], shots[sb], T, mir, m)
        if detail is not None:
            detail[('t',) + key] = t
        if m['tau_t'] is None or t is None or t >= m['tau_t']:
            out[key] = n
    return out, n_ver


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def cluster_shots(fps_, m=M):
    score, n_ver = match_shots(fps_, m)
    uf = UF(sum(len(f['shots']) for f in fps_))
    for a, b in score:
        uf.union(a, b)
    labels, k = [], 0
    for f in fps_:
        labels.append([uf.find(k + i) for i in range(len(f['shots']))]); k += len(f['shots'])
    return labels, n_ver


def families(fps_, labels, p=P):
    N = len(fps_)
    df = Counter(c for L in labels for c in set(L))
    boiler = {c for c, n in df.items() if N >= 8 and n >= max(3, p['boiler'] * N)}
    dur = [defaultdict(float) for _ in range(N)]
    for fi, f in enumerate(fps_):
        for s, c in zip(f['shots'], labels[fi]):
            if c not in boiler:
                dur[fi][c] += s['t1'] - s['t0']
    inv = defaultdict(set)
    for fi in range(N):
        for c in dur[fi]:
            inv[c].add(fi)
    uf, seen = UF(N), set()
    for c, members in inv.items():
        for i in members:
            for j in members:
                if i >= j or (i, j) in seen:
                    continue
                seen.add((i, j))
                sh = set(dur[i]) & set(dur[j])
                oi = sum(dur[i][k] for k in sh) / max(sum(dur[i].values()), 1e-6)
                oj = sum(dur[j][k] for k in sh) / max(sum(dur[j].values()), 1e-6)
                if max(oi, oj) >= p['hi'] and min(oi, oj) >= p['lo']:
                    uf.union(i, j)
    fam = [uf.find(i) for i in range(N)]
    hook = []
    for fi in range(N):
        first = next((c for c in labels[fi] if c not in boiler), None)
        hook.append((fam[fi], first if first is not None else f'x{fi}'))
    return fam, hook, boiler


# ---------------------------------------------------------------- survival
def hook_hazards(rows, obs, group_of, handoff=3.0, alpha=1.0):
    """Exponential hazard per hook group with right-censoring (still running) and relaunch-aware
    censoring (an ID that ends as a sibling ID of the same hook starts is a handoff, not a kill).
    Gamma(alpha, beta) prior centred on the pooled hazard shrinks thin evidence."""
    by = defaultdict(list)
    for r in rows:
        by[group_of[r['file']]].append(r)
    st = {}
    for g, rs in by.items():
        E = D = 0.0
        for r in rs:
            end = obs if r['end'] is None else r['end']
            E += max(end - r['start'], 0.5)
            if r['end'] is not None and not any(
                    o is not r and o['start'] >= r['start'] and abs(o['start'] - r['end']) <= handoff for o in rs):
                D += 1
        st[g] = (E, D, len(rs))
    tE, tD = sum(v[0] for v in st.values()), sum(v[1] for v in st.values())
    beta = alpha / ((tD + 1) / (tE + 1))
    return {g: dict(exposure=E, kills=D, ids=n, hazard=(D + alpha) / (E + beta)) for g, (E, D, n) in st.items()}


# ---------------------------------------------------------------- report helpers
def frame_b64(path, t, h=150):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return ''
    f = cv2.resize(f, (round(h * f.shape[1] / f.shape[0]), h))
    return base64.b64encode(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()


def analyze(fps_, rows=None, obs=None):
    t = time.time()
    labels, n_ver = cluster_shots(fps_)
    fam, hook, boiler = families(fps_, labels)
    out = dict(labels=labels, fam=fam, hook=hook, boiler=sorted(boiler), n_verified=n_ver,
               match_seconds=time.time() - t)
    if rows is not None:
        g = {f['path']: hook[i] for i, f in enumerate(fps_)}
        out['hazards'] = hook_hazards(rows, obs, g)
    return out


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('csv'); ap.add_argument('--obs', required=True); ap.add_argument('--out', default='report.html')
    a = ap.parse_args()
    day = lambda s: (dt.date.fromisoformat(s) - dt.date(2000, 1, 1)).days
    rows = [dict(ad_id=r['ad_id'], file=r['file'], start=day(r['start']), end=day(r['end']) if r['end'].strip() else None)
            for r in csv.DictReader(open(a.csv, newline='', encoding='utf-8'))]
    files = sorted({r['file'] for r in rows})
    fps_ = [fingerprint(f) for f in files]
    res = analyze(fps_, rows, day(a.obs))
    hz = res['hazards']
    by_fam = defaultdict(list)
    for g, v in hz.items():
        by_fam[g[0]].append((v['hazard'], g, v))
    parts = ['<h1>recut report</h1>']
    for f, hs in sorted(by_fam.items(), key=lambda kv: -sum(x[2]['ids'] for x in kv[1])):
        members = [i for i, x in enumerate(res['fam']) if x == f]
        parts.append(f'<h2>Family: {len(members)} creatives, {sum(x[2]["ids"] for x in hs)} ad IDs</h2><div class=row>')
        for rank, (h, g, v) in enumerate(sorted(hs, key=lambda x: x[0])):
            i = next(i for i in members if res['hook'][i] == g)
            img = frame_b64(files[i], fps_[i]['shots'][0]['t1'] / 2)
            parts.append(f'<div class=card><img src="data:image/jpeg;base64,{img}"><b>{"likely winner" if rank == 0 and len(hs) > 1 else ""}</b>'
                         f'<p>{v["ids"]} IDs, {v["exposure"]:.0f} ad-days, {v["kills"]:.0f} kills<br>hazard {v["hazard"]:.3f}/day</p></div>')
        parts.append('</div>')
    css = 'body{font:14px system-ui;margin:24px}.row{display:flex;gap:12px;flex-wrap:wrap}.card{border:1px solid #ccc;padding:8px;border-radius:8px}'
    open(a.out, 'w', encoding='utf-8').write(f'<!doctype html><meta charset=utf-8><style>{css}</style>' + ''.join(parts))
    print(f'{len(files)} creatives, {len(set(res["fam"]))} families -> {a.out}')


if __name__ == '__main__':
    _cli()
