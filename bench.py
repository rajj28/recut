"""Benchmark for recut. Builds a competitor ad set with known ground truth from CC-BY film footage
(dev = Elephants Dream, test = Tears of Steel; (c) Blender Foundation, CC-BY), so thresholds are
tuned on one film and scored on another.

Each family = 3 body shots + one of 2-4 hooks (hooks are reused ACROSS families, as UGC openers are).
Every creative is a recut: 9:16 / 1:1 / 4:5 / 16:9 crops with random reframing, blurred-bar 9:16
padding, zoom up to 15%, mirroring, burned-in captions that change every 1.2 s, trims, CRF 23-33,
two resolutions, some body iterations (a shot dropped), and a shared brand end card with varying offers.

  python bench.py make dev|test      render the ad set (~2-4 min)
  python bench.py run  dev|test      fingerprint, group, score, simulate launches -> results_<set>.json
  python bench.py report             -> report.html
"""
import html, json, pickle, random, statistics, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import recut

ROOT = Path(__file__).parent / 'bench_data'
SRC = {'dev': ROOT / 'src' / 'ed.mov', 'test': ROOT / 'src' / 'tos.mov', 'test2': ROOT / 'src' / 'tos.mov', 'test3': ROOT / 'src' / 'tos.mov'}
SEED = {'dev': 11, 'test': 22, 'test2': 33, 'test3': 44}
COLOR = {'dev': (200, 90, 40), 'test': (40, 60, 190), 'test2': (60, 170, 60), 'test3': (170, 60, 170)}
FR = 25
PHRASES = ["I wish I knew this sooner", "No fees. Seriously.", "Here's how it works", "Stop overpaying",
           "This changed my mornings", "POV: payday came early", "3 things nobody tells you", "Wait for it",
           "My honest review", "Made for people like you", "Try it free today", "Watch till the end"]
OFFERS = ["Get $25 when you join", "0% fees for 12 months", "Up to 4.6% APY", "Free for 30 days", "Get $300 bonus"]
AR = {'16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1.0, '4:5': 0.8, '9:16pad': 9 / 16}
OUT_H = {'16:9': 360, '9:16': 640, '1:1': 480, '4:5': 600, '9:16pad': 640}
even = lambda x: max(2, int(round(x / 2)) * 2)


# ---------------------------------------------------------------- make
def read_frames(path):
    cap, fs = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f)
    cap.release()
    return fs


def build_library(name, n, rng):
    src = SRC[name]
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(src)],
                               capture_output=True, text=True).stdout)
    lib = ROOT / name / 'lib'
    lib.mkdir(parents=True, exist_ok=True)
    chosen, sigs, tries = [], [], 0
    while len(chosen) < n and tries < n * 12:
        tries += 1
        t = rng.uniform(25, dur - 150)
        if any(abs(t - c['t']) < 4.0 for c in chosen):
            continue
        out = lib / f's{len(chosen):03d}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.2f}', '-i', str(src), '-t', '3.4', '-vf',
                        f'scale=-2:360,fps={FR}', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14', str(out)],
                       check=True)
        fs = read_frames(out)
        if len(fs) < 80:
            continue
        g = [cv2.cvtColor(fs[i], cv2.COLOR_BGR2GRAY) for i in (5, len(fs) // 2, len(fs) - 6)]
        if min(x.std() for x in g) < 20:          # skip black / flat / fade footage
            continue
        sig = cv2.resize(g[1], (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        sig = (sig - sig.mean()) / (sig.std() + 1e-6)
        if sigs and max(float(sig @ s) / len(sig) for s in sigs) > 0.7:   # ground-truth hygiene
            continue
        sigs.append(sig)
        chosen.append(dict(path=str(out), t=round(t, 2)))
    if len(chosen) < n:
        sys.exit(f'only {len(chosen)}/{n} usable segments')
    return chosen


def design(name, rng, lib, F=12, S=6, Pn=18):
    body, single, pool = lib[:F * 3], lib[F * 3:F * 3 + S * 3], lib[F * 3 + S * 3:F * 3 + S * 3 + Pn]
    files = []

    def spec(**kw):
        return dict(kw, ar=rng.choices(list(AR), [0.15, 0.35, 0.2, 0.15, 0.15])[0],
                    zoom=1.0 if rng.random() < 0.5 else round(rng.uniform(1.03, 1.15), 3), flip=rng.random() < 0.1,
                    cap=rng.random() < 0.85, cap_pos=rng.choice(['bottom', 'center', 'top']), crf=rng.choice([23, 28, 33]),
                    res=rng.choice([1.0, 1.5]), card=rng.random() < 0.75, offer=rng.choice(OFFERS), seed=rng.random())

    for f in range(F):
        for hi, hp in enumerate(rng.sample(range(Pn), rng.choice([2, 3, 3, 4]))):
            for _ in range(rng.choice([1, 2, 2, 3])):
                files.append(spec(family=f'F{f}', hook=f'F{f}H{hi}', hook_seg=hp,
                                  segs=[pool[hp]['path']] + [b['path'] for b in body[3 * f:3 * f + 3]]))
        if rng.random() < 0.35:
            base, drop = rng.choice([x for x in files if x['family'] == f'F{f}']), rng.randrange(3)
            files.append(spec(family=base['family'], hook=base['hook'], hook_seg=base['hook_seg'], iteration=True,
                              segs=[base['segs'][0]] + [s for i, s in enumerate(base['segs'][1:]) if i != drop]))
    for s in range(S):
        files.append(spec(family=f'S{s}', hook=f'S{s}H0', hook_seg=None, segs=[x['path'] for x in single[3 * s:3 * s + 3]]))
    rng.shuffle(files)
    (ROOT / name / 'ads').mkdir(parents=True, exist_ok=True)
    for i, x in enumerate(files):
        x['file'] = str(ROOT / name / 'ads' / f'ad{i:03d}.mp4')
    return files


def transform(frame, sp, off, W, H):
    h, w = frame.shape[:2]
    if sp['zoom'] > 1:
        cw, ch = int(w / sp['zoom']), int(h / sp['zoom'])
        frame = cv2.resize(frame[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw], (w, h))
    if sp['flip']:
        frame = frame[:, ::-1]
    if sp['ar'] == '9:16pad':
        ch = even(W * h / w)
        bw = even(H * w / h)
        bg = cv2.resize(frame, (bw, H))[:, (bw - W) // 2:(bw - W) // 2 + W]
        bg = (cv2.GaussianBlur(bg, (0, 0), 18) * 0.55).astype(np.uint8)
        y = (H - ch) // 2
        bg[y:y + ch] = cv2.resize(np.ascontiguousarray(frame), (W, ch))
        return bg
    cw = min(w, int(h * AR[sp['ar']]))
    x = int((w - cw) * off)
    return cv2.resize(np.ascontiguousarray(frame[:, x:x + cw]), (W, H))


def put_center(img, text, y, scale, color=(255, 255, 255), outline=True):
    H, W = img.shape[:2]
    font, th = cv2.FONT_HERSHEY_DUPLEX, max(2, int(scale * 2))
    tw = cv2.getTextSize(text, font, scale, th)[0][0]
    if tw > 0.9 * W:
        scale *= 0.9 * W / tw
        th = max(2, int(scale * 2))
        tw = cv2.getTextSize(text, font, scale, th)[0][0]
    if outline:
        cv2.putText(img, text, ((W - tw) // 2, y), font, scale, (0, 0, 0), th + 4, cv2.LINE_AA)
    cv2.putText(img, text, ((W - tw) // 2, y), font, scale, color, th, cv2.LINE_AA)


def end_card(W, H, offer, color):
    img = (np.array(color, np.float32)[None, None] * np.linspace(0.7, 1.0, H)[:, None, None]).astype(np.uint8)
    img = np.ascontiguousarray(np.broadcast_to(img, (H, W, 3)))
    put_center(img, 'NOVA', int(H * 0.45), W / 360 * 2.4, outline=False)
    put_center(img, offer, int(H * 0.58), W / 360 * 0.8, outline=False)
    put_center(img, 'Download now', int(H * 0.68), W / 360 * 0.7, (230, 230, 230), outline=False)
    return img


def render(sp, color):
    h = even(OUT_H[sp['ar']] * sp['res'])
    W, H = even(h * AR[sp['ar']]), h
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FR),
                          '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(sp['crf']), '-pix_fmt', 'yuv420p',
                          sp['file']], stdin=subprocess.PIPE)
    r = random.Random(sp['seed'])
    phrases = r.sample(PHRASES, len(PHRASES))
    pos = sp['cap_pos'] if sp['ar'] != '9:16pad' else ('top' if sp['cap_pos'] == 'top' else 'bottom')
    ypos = ({'top': 0.11, 'bottom': 0.91} if sp['ar'] == '9:16pad' else {'bottom': 0.82, 'center': 0.52, 'top': 0.16})[pos]
    n, cuts, trims = 0, [], []
    for si, seg in enumerate(sp['segs']):
        fs = read_frames(seg)
        a, b = r.randint(0, 7), len(fs) - r.randint(0, 7)
        if si == 0:
            b = min(b, a + r.randint(50, 65))           # hooks run 2.0-2.6 s
        # editors reframe around the subject: fixed per shot, small jitter per recut
        subj = 0.3 + 0.4 * (sum(map(ord, Path(seg).stem)) * 7919 % 1000) / 1000
        off = min(1.0, max(0.0, subj + r.uniform(-0.06, 0.06)))
        trims.append(a)
        if si:
            cuts.append(round(n / FR, 2))
        for f in fs[a:b]:
            img = transform(f, sp, off, W, H)
            if sp['cap']:
                put_center(img, phrases[(n // 30) % len(phrases)], int(H * ypos), W / 360 * 0.85)
            p.stdin.write(img.tobytes())
            n += 1
    if sp['card']:
        cuts.append(round(n / FR, 2))
        card = end_card(W, H, sp['offer'], color).tobytes()
        for _ in range(50):
            p.stdin.write(card)
            n += 1
    p.stdin.close()
    p.wait()
    sp.update(cuts=cuts, trims=trims, dur=round(n / FR, 2), size=[W, H])


def make(name):
    rng = random.Random(SEED[name])
    lib = build_library(name, 12 * 3 + 6 * 3 + 18, rng)
    files = design(name, rng, lib)
    t = time.time()
    for sp in files:
        render(sp, COLOR[name])
    json.dump(dict(files=files, lib=lib), open(ROOT / name / 'gt.json', 'w'), indent=1)
    print(f'{name}: {len(files)} creatives, {len({x["family"] for x in files})} families, rendered in {time.time() - t:.0f}s')


# ---------------------------------------------------------------- baselines + metrics
def dhash(path, box=None):
    fr = recut.decode(path, fps=2)
    if box:
        fr = fr[:, box[0]:box[1], box[2]:box[3]]
    bits = np.stack([(lambda s: (s[:, 1:] > s[:, :-1]).ravel())(cv2.resize(f, (9, 8), interpolation=cv2.INTER_AREA)) for f in fr])
    return np.packbits(bits, axis=1)                 # n x 8 bytes


def hash_groups(H, thr=10, link=0.4):
    N, uf = len(H), recut.UF(len(H))
    for i in range(N):
        for j in range(i + 1, N):
            x = np.bitwise_xor(H[i][:, None, :], H[j][None, :, :])
            d = np.unpackbits(x, axis=2).sum(2)
            if max((d.min(1) <= thr).mean(), (d.min(0) <= thr).mean()) >= link:
                uf.union(i, j)
    return [uf.find(i) for i in range(N)]


def pair_prf(true, pred, mask=None):
    tp = fp = fn = 0
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            if mask and not mask(i, j):
                continue
            t, p = true[i] == true[j], pred[i] == pred[j]
            tp += t and p
            fp += p and not t
            fn += t and not p
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(P=round(P, 3), R=round(R, 3), F1=round(2 * P * R / max(P + R, 1e-9), 3), pairs=tp + fn)


# ---------------------------------------------------------------- launch simulation
def simulate(fam_hooks, files_of, q, rng, T=90, sigma=2.5, theta=-0.2, p_dup=0.06, p_pause=0.5, budget=0.01,
             launch_max=80, cap=10, rule='cum'):
    """A media buyer: daily noisy performance around the hook's true quality; after a 3-day learning
    phase, kill when the cumulative mean (rule='cum') or the last-3-day mean (rule='roll') drops below theta; random budget stops; survivors get duplicated into new ad sets (often a recut),
    and the original is paused at that moment with probability p_pause."""
    rows = []
    for f, hooks in fam_hooks.items():
        L, cnt = rng.uniform(0, launch_max), Counter()
        queue = [(h, L + rng.uniform(0, 2), None) for h in hooks]
        while queue:
            h, s, pf = queue.pop()
            if s >= T or cnt[h] >= cap:
                continue
            cnt[h] += 1
            fl = pf if (pf is not None and rng.random() < 0.5) else rng.choice(files_of[h])
            perf, end, day = [], None, s
            while day < T:
                perf.append(rng.gauss(q[h], sigma))
                age = day - s
                win = perf if rule == 'cum' else perf[-3:]
                if age >= 3 and sum(win) / len(win) < theta or rng.random() < budget:
                    end = day
                    break
                if age >= 5 and rng.random() < p_dup:
                    queue.append((h, day + 1, fl))
                    if rng.random() < p_pause:
                        end = day + 1
                        break
                day += 1
            rows.append(dict(file=fl, hook=h, family=f, start=s, end=end))
    return rows


def winner_eval(files, pred_fam, pred_hook, n_sims=300, seed=5, **kw):
    T = 90
    hook_of = {x['file']: x['hook'] for x in files}
    fam_hooks, files_of = defaultdict(set), defaultdict(list)
    for x in files:
        fam_hooks[x['family']].add(x['hook'])
        files_of[x['hook']].append(x['file'])
    fam_hooks = {f: sorted(h) for f, h in fam_hooks.items()}
    pf = {x['file']: pred_fam[i] for i, x in enumerate(files)}
    ph = {x['file']: pred_hook[i] for i, x in enumerate(files)}
    rng = random.Random(seed)
    score, n = Counter(), 0
    span = lambda rs: max((T if r['end'] is None else r['end']) for r in rs) - min(r['start'] for r in rs)
    for _ in range(n_sims):
        q = {h: rng.gauss(0, 1) for hs in fam_hooks.values() for h in hs}
        rows = simulate(fam_hooks, files_of, q, rng, T=T, **kw)
        by_fam = defaultdict(list)
        for r in rows:
            by_fam[r['family']].append(r)
        for f, hooks in fam_hooks.items():
            if len(hooks) < 2:
                continue
            n += 1
            best = max(hooks, key=q.get)
            rs = by_fam[f]
            score['chance'] += 1 / len(hooks)
            longest = max(rs, key=lambda r: (T if r['end'] is None else r['end']) - r['start'])
            score['M1 longest single ad ID'] += longest['hook'] == best
            byfile, byhook = defaultdict(list), defaultdict(list)
            for r in rs:
                byfile[r['file']].append(r)
                byhook[r['hook']].append(r)
            score['M2 longest creative span (exact dupes merged)'] += hook_of[max(byfile, key=lambda k: span(byfile[k]))] == best
            score['M3 longest hook span (recuts grouped, oracle)'] += max(byhook, key=lambda k: span(byhook[k])) == best
            hz = recut.hook_hazards(rs, T, hook_of)
            score['M4 censored hazard (recuts grouped, oracle)'] += min(hz, key=lambda g: hz[g]['hazard']) == best
            # end to end: recut's own grouping
            fam_pred = Counter(pf[r['file']] for r in rs).most_common(1)[0][0]
            rp = [r for r in rows if pf[r['file']] == fam_pred]
            hz = recut.hook_hazards(rp, T, ph)
            g = min(hz, key=lambda g: hz[g]['hazard'])
            score['M5 recut end to end (own grouping + hazard)'] += Counter(r['hook'] for r in rp if ph[r['file']] == g).most_common(1)[0][0] == best
    return {k: round(v / n, 3) for k, v in sorted(score.items())}, n


def winner_sweep(files, pred_fam, pred_hook):
    res = {}
    for rule in ('cum', 'roll'):
        for pp in (0.0, 0.5, 0.9):
            acc, n = winner_eval(files, pred_fam, pred_hook, p_pause=pp, rule=rule)
            res[f'{rule} p_pause={pp}'] = acc
    res['family_trials_per_setting'] = n
    return res


def winners(name):
    """recompute only the winner-pick simulation from saved predictions"""
    p = ROOT.parent / f'results_{name}.json'
    R = json.load(open(p))
    files = json.load(open(ROOT / name / 'gt.json'))['files']
    R['winner_pick'] = winner_sweep(files, R['pred']['fam'], R['pred']['hook'])
    json.dump(R, open(p, 'w'), indent=1)
    for k, v in R['winner_pick'].items():
        print(k, v)


def run(name):
    gt = json.load(open(ROOT / name / 'gt.json'))
    files = gt['files']
    cache = ROOT / name / 'fp.pkl'
    if cache.exists():
        fps_, t_fp = pickle.load(open(cache, 'rb'))
    else:
        t = time.time()
        fps_ = [recut.fingerprint(x['file']) for x in files]
        t_fp = time.time() - t
        pickle.dump((fps_, t_fp), open(cache, 'wb'))
    print(f'fingerprint {t_fp:.0f}s, shots {sum(len(f["shots"]) for f in fps_)}', flush=True)
    res = recut.analyze(fps_)
    print(f'match {res["match_seconds"]:.0f}s, verified {res["n_verified"]}', flush=True)
    video_s = sum(f['dur'] for f in fps_)
    t = time.time()
    H_plain = [dhash(x['file']) for x in files]
    H_box = [dhash(x['file'], f['box']) for x, f in zip(files, fps_)]
    g_plain, g_box = hash_groups(H_plain), hash_groups(H_box)
    t_b = time.time() - t
    tf, th = [x['family'] for x in files], [x['hook'] for x in files]
    pred_hook = [f'{a}|{b}' for a, b in res['hook']]
    masks = {'all same-family pairs': None,
             'different hook': lambda i, j: files[i]['hook'] != files[j]['hook'],
             'different aspect ratio': lambda i, j: files[i]['ar'] != files[j]['ar'],
             'one is mirrored': lambda i, j: files[i]['flip'] != files[j]['flip'],
             'one is zoomed': lambda i, j: (files[i]['zoom'] > 1) != (files[j]['zoom'] > 1)}
    recall = {k: {m: pair_prf(tf, g, mk)['R'] for m, g in (('frame hash', g_plain), ('frame hash + bar strip', g_box), ('recut', res['fam']))}
              | {'pairs': pair_prf(tf, tf, mk)['pairs']} for k, mk in masks.items()}
    hit = tot = 0
    for x, f in zip(files, fps_):
        det = [s['t0'] for s in f['shots'][1:]]
        for c in x['cuts']:
            tot += 1
            hit += any(abs(c - d) <= 0.25 for d in det)
    out = dict(
        set=name, creatives=len(files), families=len(set(tf)), video_minutes=round(video_s / 60, 1),
        family_grouping={'frame hash': pair_prf(tf, g_plain), 'frame hash + bar strip': pair_prf(tf, g_box),
                         'recut': pair_prf(tf, res['fam'])},
        hook_grouping={'recut': pair_prf(th, pred_hook), 'frame hash': pair_prf(th, g_plain)},
        recall_by_condition=recall, cut_recall=round(hit / tot, 3),
        speed=dict(fingerprint_x_realtime=round(video_s / t_fp, 1), match_seconds=round(res['match_seconds'], 1),
                   verified_shot_pairs=res['n_verified'], baseline_seconds=round(t_b, 1)),
        pred=dict(fam=[str(v) for v in res['fam']], hook=pred_hook, frame_hash=[str(v) for v in g_plain]))
    out['winner_pick'] = winner_sweep(files, out['pred']['fam'], pred_hook)
    json.dump(out, open(ROOT.parent / f'results_{name}.json', 'w'), indent=1)
    show = {k: v for k, v in out.items() if k != 'pred'}
    print(json.dumps(show, indent=1))

def report(*names):
    """report.html: metrics for each set + one family that frame hashing shatters, shown frame by frame."""
    esc = html.escape
    def table(rows, head):
        h = ''.join(f'<th>{esc(str(c))}</th>' for c in head)
        b = ''.join('<tr>' + ''.join(f'<td>{esc(str(c))}</td>' for c in r) + '</tr>' for r in rows)
        return f'<table><tr>{h}</tr>{b}</table>'
    parts = []
    for name in names:
        R = json.load(open(ROOT.parent / f'results_{name}.json'))
        files = json.load(open(ROOT / name / 'gt.json'))['files']
        film = {'dev': 'Elephants Dream, used for tuning', 'test': 'Tears of Steel draw 1: exposed the subtitle failure, so no longer clean', 'test2': 'Tears of Steel draw 2: exposed the same-set failure, so no longer clean', 'test3': 'Tears of Steel draw 3: fresh held-out, scored once'}[name]
        parts.append(f'<h2>{esc(name)} set: {esc(film)}, {R["creatives"]} creatives, {R["families"]} families, {R["video_minutes"]} min</h2>')
        parts.append('<h3>Grouping recuts into families (pairwise)</h3>' + table(
            [[k, v['P'], v['R'], v['F1']] for k, v in R['family_grouping'].items()], ['method', 'precision', 'recall', 'F1']))
        parts.append('<h3>Grouping by hook</h3>' + table(
            [[k, v['P'], v['R'], v['F1']] for k, v in R['hook_grouping'].items()], ['method', 'precision', 'recall', 'F1']))
        rc = R['recall_by_condition']
        meth = ['frame hash', 'frame hash + bar strip', 'recut']
        parts.append('<h3>Recall on same-family pairs, by what changed</h3>' + table(
            [[k, v['pairs']] + [v[m] for m in meth] for k, v in rc.items()], ['pair type', 'pairs'] + meth))
        wp = R['winner_pick']
        keys = [k for k in wp if 'p_pause' in k]
        ms = list(wp[keys[0]])
        parts.append(f'<h3>Picking the best hook in a family (simulated media buyer, {wp["family_trials_per_setting"]} family trials per setting)</h3>'
                     '<p>cum = buyer kills on cumulative performance; roll = on the last 3 days (noisier). p_pause = chance the original is paused when a winner is duplicated (0.9 = winners keep getting relaunched under new IDs). M1-M4 are given the true families, which favours them.</p>'
                     + table([[m] + [wp[k][m] for k in keys] for m in ms], ['method'] + keys))
        sp = R['speed']
        parts.append(f'<p>Speed: fingerprinting {sp["fingerprint_x_realtime"]}x real time, matching {sp["match_seconds"]} s '
                     f'({sp["verified_shot_pairs"]} keyframe pairs verified). Cut recall {R["cut_recall"]}.</p>')
        # example: family that frame hash splits the most while recut keeps it whole and clean
        pf, ph = R['pred']['fam'], R['pred']['frame_hash']
        best = None
        for fam in sorted({x['family'] for x in files if x['family'].startswith('F')}):
            ix = [i for i, x in enumerate(files) if x['family'] == fam]
            clean = len({pf[i] for i in ix}) == 1 and sum(pf[j] == pf[ix[0]] for j in range(len(files))) == len(ix)
            k = len({ph[i] for i in ix})
            if clean and (best is None or k > best[1]):
                best = (fam, k, ix)
        if best:
            fam, k, ix = best
            parts.append(f'<h3>Example: family {fam}. Frame hashing split it into {k} groups; recut found 1</h3><div class="grid">')
            for i in ix:
                x = files[i]
                t_body = (x['cuts'][0] + 1.0) if x['cuts'] else 1.0
                tag = f'{x["hook"]} | {x["ar"]}{" mirrored" if x["flip"] else ""}{" zoom %.2f" % x["zoom"] if x["zoom"] > 1 else ""} | hash group {ph[i]}'
                parts.append(f'<figure><img src="data:image/jpeg;base64,{recut.frame_b64(x["file"], 0.6, 120)}">'
                             f'<img src="data:image/jpeg;base64,{recut.frame_b64(x["file"], t_body, 120)}">'
                             f'<figcaption>{esc(tag)}</figcaption></figure>')
            parts.append('</div><p>Left: hook frame. Right: body frame.</p>')
    css = ('body{font:14px system-ui;margin:24px;max-width:1100px}table{border-collapse:collapse;margin:6px 0 14px}'
           'td,th{border:1px solid #ccc;padding:3px 8px;text-align:left}th{background:#f3f3f3}'
           '.grid{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;border:1px solid #ddd;padding:4px}'
           'figure img{height:120px;margin-right:2px}figcaption{font-size:12px}')
    out = ROOT.parent / 'report.html'
    out.write_text(f'<!doctype html><meta charset="utf-8"><title>recut benchmark</title><style>{css}</style>'
                   '<h1>recut benchmark</h1>' + ''.join(parts), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    {'make': make, 'run': run, 'report': report, 'winners': winners}[sys.argv[1]](*sys.argv[2:])
