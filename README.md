# recut

**Find which hook is winning in a competitor's video ads, using only what an ad library shows.**

Ad libraries show the creative, the start date and whether the ad is still running. They show no spend and no CTR. The usual winner proxy is days active, but it only works if you know which ads are really the same ad. Winners come back as recuts: a new 3-second hook, a 9:16 or 1:1 crop, a zoom, a mirror, new burned-in captions and a different end-card offer. Frame hashes treat each recut as an unrelated creative.

recut groups recuts into **variant families** by matching the footage itself. Within each family, the shots that differ are the hooks being tested. A censoring-aware survival estimate then ranks the hooks.

**Visual report with example families and thumbnails: https://rajj28.github.io/recut/report.html**

## Results

The benchmark has known ground truth, built from CC-BY Blender films (details below).
- Thresholds were tuned on **Elephants Dream** only.
- Scoring was done on **Tears of Steel**, a different film. Each test draw is a fresh random set of about 72 creatives in 18 families.

**Grouping recuts into families (pairwise precision / recall / F1)**

| set | frame hash (dHash) | recut |
|---|---|---|
| dev, Elephants Dream (tuning) | 1.00 / 0.06 / 0.12 | 1.00 / 0.94 / **0.97** |
| test draw 1, Tears of Steel | 1.00 / 0.08 / 0.14 | 0.79 / 0.95 / **0.86** |
| test draw 2, Tears of Steel | 0.91 / 0.06 / 0.12 | 1.00 / 1.00 / **1.00** |
| **test draw 3, Tears of Steel (fresh, scored once)** | 1.00 / 0.04 / 0.08 | 1.00 / 1.00 / **1.00** |

Draws 1 and 2 each exposed a failure, and I fixed both (see "How it got here"), so they are no longer clean held-out sets. Draw 3 was generated after the last change and scored once.

**Share of same-family pairs found, by what changed (test draw 3)**

| pair type | pairs | frame hash | recut |
|---|---|---|---|
| different hook | 141 | 0.04 | 1.00 |
| different aspect ratio | 142 | 0.00 | 1.00 |
| one is mirrored | 23 | 0.00 | 1.00 |
| one is zoomed | 89 | 0.07 | 1.00 |

- **Grouping by hook (draw 3):** precision 1.00, recall 0.91, F1 0.95. Frame hashing gets F1 0.12.
- **Cut detection recall:** 0.996.
- **Speed:** about 16x real time to fingerprint on one laptop CPU process. Matching took 47 s for 73 creatives (15 min of video).

**Picking the best hook in a family.** This uses a simulated media buyer:
- Daily performance is noisy. Losers are killed after a learning phase.
- Winners get duplicated, and the original is paused with probability p_pause.
- Results are from 3,600 family trials per setting, on draw 3.

| method | cumulative kill rule, p_pause 0 | 0.5 | 0.9 |
|---|---|---|---|
| chance | 0.36 | 0.36 | 0.36 |
| M1 longest-running single ad ID | 0.60 | 0.60 | 0.60 |
| M3 longest hook span (true families given) | 0.60 | 0.61 | 0.61 |
| M4 censored hazard (true families given) | 0.62 | 0.63 | 0.61 |
| M5 recut end to end (its own grouping) | 0.62 | 0.62 | 0.61 |

**What this does and doesn't show.**
- Once ads are grouped, survival finds the best hook about 60% of the time, against 36% for chance.
- Handling censoring and relaunches adds only about 2 points over "longest-running ID", and only when that baseline is also given the true families.
- Under a noisier kill rule (last 3 days instead of cumulative), every method sits at about 0.58.
- So the value is in the grouping. Without it, the question "which hook won in this concept" can't even be asked, because 15 recuts look like 15 unrelated ads. The survival signal on its own is weak.

## How it works

```
decode 10 fps -> content box: strip black/blurred bars (low Laplacian energy, symmetric pairs)
-> shot cuts: 4x4 tile differences, 40th percentile, so a caption swap is not a cut
-> keyframes: SIFT on the content crop, scaled to a fixed height (crops keep the full source height,
   so after this only zoom changes scale), plus the mirror image
-> one approximate-NN index (FLANN) over every descriptor: no all-pairs comparison
-> putative matches grouped by keyframe pair -> RANSAC similarity transform; accept only:
     no rotation, scale 0.7-1.45 (zoom range), >= 12 inliers,
     inliers covering >= 2 rows and 2 columns, and spread vertically (captions sit in one band)
-> same-clip check: warp shot B onto A, correlate frame-difference images at the best time offset
   (the same clip moves the same way; another take filmed on the same set does not)
-> shot clusters (union-find) -> drop clusters in >= 25% of ads (brand end cards)
-> link ads that share >= 50% / >= 30% of their non-boilerplate duration -> families
-> hook = first non-boilerplate shot
-> per-hook exponential hazard, with still-running ads right-censored, an ID that ends within
   3 days of a sibling's start treated as a handoff (not a kill), and a Gamma prior
```

## How it got here (including what failed)

The scores before each fix are kept in `results_dev_v1.json`, `results_test_v1.json` and `results_test2_v2.json`.

1. **First version: pixel correlation of small thumbnails.** On dev it found 67% of true shot matches but also flagged 3% of unrelated pairs. Burned-in caption text and dark scenes produced false matches. I replaced it with SIFT + RANSAC: 83% recall at 0.04% false positives on the same sampled pairs. The fix that mattered most was normalizing every crop to the same height, because a scale bound had been rejecting 9:16 vs 16:9 matches.
2. **Test draw 1 scored precision 0.72.** I first guessed burned-in subtitles and added the vertical-spread gate, calibrated on dev. It changed nothing (0.70), so that guess was wrong.
3. **Test draw 2 scored precision 0.47**, and its false links showed the real cause: 98% joined segments within 60 s of each other in the film, and none involved the end card. That's the same dialogue scene, with the same camera setup recurring. The footage really does match; it's just a different moment. This is a real case too: a UGC creator who records ten scripts against one wall. I added the same-clip motion check, set its threshold by a dev-only rule (keep 97.5% of true dev matches), and scored a fresh draw 3 once. Draw 2 rescored with the final method: 1.00. Draw 1: 0.79.

## Limits

- **Synthetic benchmark.** Two films and one ad generator. Real ad sets (UGC talking heads, product demos, text-heavy motion graphics) haven't been tested yet. Motion-graphics ads with little camera footage are the likely weak spot.
- **Static shots.** 34% of true dev matches have too little motion for the same-clip check, so they rely on the spatial match alone. Two takes of a static shot on the same set will merge.
- **Draw 1 still has 0.79 precision.** The remaining errors are same-scene segments whose motion correlates.
- **One concept, one clip.** If two concepts share an identical clip, recut puts them in one family. That's by design.
- **Simulated buyer.** The media-buyer policy is an assumption, not data. Treat the winner-pick numbers as a sanity check, not a measurement.
- **Scale.** The index lives in memory. Families exist within one advertiser, so matching per advertiser keeps it to hundreds or thousands of ads. Cross-advertiser matching at millions of ads would need FAISS IVF-PQ and sharding.
- **Not covered:** shifts in messaging over time (OCR/ASR plus an LLM would be the next module), and hooks that differ only by text overlay on the same footage.

## Run it

Needs Python 3.10+, `pip install opencv-python numpy` (tested with OpenCV 5.0, NumPy 2.4), and ffmpeg/ffprobe on PATH.

**On real ads.**
1. Save each creative's video from the Meta Ad Library by hand, along with its library ID and dates. Don't scrape.
2. Write a CSV with the columns `ad_id,file,start,end` (dates as YYYY-MM-DD; leave `end` empty if the ad is still running).
3. Run:

```
python recut.py ads.csv --obs 2026-11-30 --out report.html
```

The report shows each family, its hooks with thumbnails, ad-ID counts, exposure and hazard, and the likely winner.

**Benchmark.** Put `ed.mov` (Elephants Dream) and `tos.mov` (Tears of Steel), both from download.blender.org, in `bench_data/src/`. Then:

```
python bench.py make dev      # render ~86 creatives with known families (~5 min)
python bench.py run dev       # fingerprint, group, score, simulate -> results_dev.json
python bench.py make test3 && python bench.py run test3
python bench.py report dev test test2 test3    # -> report.html
```

Footage: Elephants Dream and Tears of Steel, (c) Blender Foundation, CC-BY.
