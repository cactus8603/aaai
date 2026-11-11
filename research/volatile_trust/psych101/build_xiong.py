"""Build per-participant / per-game episodes for Psych-101 xiong2023neural/exp1 ("Changing bandit").

Sources (see .cache/datasets/psych101_xiong2023/provenance.json):
  * Psych-101 text slice (30 participants, 6 games each): instruction wording, machine letters, human choices.
  * Raw parquet (marcelbinz/xiong2023neural): true per-trial values slot1_value/slot2_value, hazard rate, RT.
The two sources are cross-checked trial by trial (choice letter <-> raw choice, points <-> raw reward).

Run with an isolated interpreter (the inputs are untrusted downloaded data):
    .venv/bin/python -I research/volatile_trust/psych101/build_xiong.py

MODEL PROTOCOL (documented in PROTOCOL.md next to this file)
  * Episodes = the three 100-trial games per participant (h = 0.1, 0.2, 0.4; Psych-101 games 1/3/5).
    Each 100-trial game directly precedes a 1500-trial game with the same h, so at games 3/5 the human
    has already played 1,600/3,200 trials; h order is counterbalanced (6 orders). `position` (1/2/3 =
    game 1/3/5) is stored for covariate analysis. 30 participants x 3 games x 100 trials = 9000 queries.
    The 1500-trial games are written to episodes_long.jsonl for human-only reference fits (not queried).
  * Teacher-forced replay: at trial t the model sees the instructions, the game header and the HUMAN's
    trials 1..t-1 (choice + points received), then is asked which machine it presses on trial t.
    P(machine) comes from the raw first-token logits of the two machine letters. Model and human are
    therefore compared on identical evidence (Psych-101 / Centaur practice).
  * Each game is shown as a standalone game ("Game 1."); the model never sees the participant's other games.
  * Machine letters are the participant's own Psych-101 letters (random per participant). All 26 uppercase
    letters are single tokens for every local Qwen2.5-VL / Qwen3-VL / InternVL3.5 tokenizer, so no remap is
    needed; build asserts this mapping table (LABEL_REMAP) is empty and records it.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / '.cache/datasets/psych101_xiong2023/raw'
OUT = Path(__file__).resolve().parent / 'data'
LABEL_REMAP = {}   # letter -> single-token replacement; empty: all letters verified single-token (Qwen/InternVL)

# Exact Psych-101 instruction text (verified identical across all 30 participants up to the letters).
INSTRUCTIONS_TOLD = (
    "You are participating in multiple games involving two slot machines, labeled {A} and {B}.\n"
    "The two slot machines are different in different games.\n"
    "Each time you choose a slot machine, you get points (choosing the same slot machine will not always give you the same points).\n"
    "You select a slot machine by pressing the corresponding key.\n"
    "The expected points change randomly, abruptly, and independently with a hazard rate (which you will be told).\n"
    "When the points change, the new expected point value assigned to that slot machine is sampled from a uniform distribution (from 1 to 99 points).\n"
    "For example, if the hazard rate is 0.1, the expected points of the machines change with 10%.\n"
    "Your goal is to choose the slot machine that will give you the most points.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    prov = json.loads((HERE / 'provenance.json').read_text())
    for f in prov['files']:
        got = sha256(RAW.parent / f['local_path'])
        if got != f['sha256']:
            raise SystemExit(f"sha256 mismatch for {f['local_path']}: {got} != {f['sha256']}")
    tab = pq.read_table(RAW / 'xiong2023neural_exp1_train.parquet').to_pydict()
    raw = collections.defaultdict(list)
    for i in range(len(tab['trial'])):
        raw[(tab['participant'][i], tab['task'][i])].append({c: tab[c][i] for c in tab})
    for rows in raw.values():
        rows.sort(key=lambda r: r['trial'])
        assert [r['trial'] for r in rows] == list(range(len(rows)))

    psych = [json.loads(l) for l in open(RAW / 'psych101_xiong2023neural_exp1.jsonl')]
    short, long_, checks = [], [], collections.Counter()
    for rec in psych:
        assert rec['experiment'] == 'xiong2023neural/exp1.csv'
        pid = int(rec['participant'])
        parts = re.split(r'\n\s*\n', rec['text'])
        intro, games = parts[0], parts[1:]
        assert len(games) == 6
        m = re.match(r'You are participating in multiple games involving two slot machines, labeled (\w) and (\w)\.', intro)
        letters_intro = (m.group(1), m.group(2))
        assert intro == INSTRUCTIONS_TOLD.format(A=letters_intro[0], B=letters_intro[1]), pid
        for g, game in enumerate(games):
            head, *lines = game.split('\n')
            hm = re.fullmatch(r'Game (\d+)\. The hazard rate is ([\d.]+)\. There are (\d+) trials in this game\.', head)
            assert hm and int(hm.group(1)) == g + 1
            parsed = [re.fullmatch(r'You press <<(\w)>> and get (\d+) points\.', l) for l in lines if l.strip()]
            assert all(parsed)
            rows = raw[(pid, g)]
            assert len(rows) == len(parsed) == int(hm.group(3))
            assert abs(rows[0]['hazard_rate'] - float(hm.group(2))) < 1e-9
            letter_of = {}
            for p, r in zip(parsed, rows):
                assert int(p.group(2)) == r['reward']
                assert letter_of.setdefault(r['choice'], p.group(1)) == p.group(1)
                assert r['reward'] == (r['slot1_value'] if r['choice'] == 0 else r['slot2_value'])
            assert set(letter_of.values()) <= set(letters_intro)
            if len(letter_of) < 2:   # only one machine ever chosen in this game: infer the other from intro
                other = [x for x in letters_intro if x not in letter_of.values()][0]
                letter_of[1 - next(iter(letter_of))] = other
            labels = [LABEL_REMAP.get(letter_of[0], letter_of[0]), LABEL_REMAP.get(letter_of[1], letter_of[1])]
            checks['games_matched'] += 1
            ep = dict(episode_id=f'p{pid:02d}_g{g + 1}', participant=pid, game=g + 1, task=g,
                      hazard=float(hm.group(2)), n_trials=len(rows),
                      position=g // 2 + 1,  # 1..3: order of this h for the participant
                      labels=labels,                       # labels[k] = letter of raw choice k (slot k+1)
                      intro_letters=[LABEL_REMAP.get(x, x) for x in letters_intro],   # Psych-101 intro order
                      original_letters=[letter_of[0], letter_of[1]],
                      trials=[dict(t=i + 1, choice=int(r['choice']), reward=int(r['reward']),
                                   v=[float(r['slot1_value']), float(r['slot2_value'])],
                                   rt=float(r['RT'])) for i, r in enumerate(rows)])
            ep['change_points'] = [tr['t'] for a, tr in zip(ep['trials'], ep['trials'][1:]) if a['v'] != tr['v']]
            (short if len(rows) == 100 else long_).append(ep)

    for name, eps in (('episodes.jsonl', short), ('episodes_long.jsonl', long_)):
        with open(args.out / name, 'w') as f:
            for ep in eps:
                f.write(json.dumps(ep) + '\n')
    hz = collections.Counter(e['hazard'] for e in short)
    meta = dict(n_participants=len(psych), n_short_episodes=len(short), n_long_episodes=len(long_),
                short_episodes_per_hazard=dict(hz), checks=dict(checks), label_remap=LABEL_REMAP,
                sources={p.name: sha256(p) for p in sorted(RAW.iterdir())},
                instructions_told=INSTRUCTIONS_TOLD,
                n_change_points_per_hazard={h: sum(len(e['change_points']) for e in short if e['hazard'] == h)
                                            for h in sorted(hz)})
    (args.out / 'build_metadata.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ('instructions_told',)}, indent=1))


if __name__ == '__main__':
    main()
