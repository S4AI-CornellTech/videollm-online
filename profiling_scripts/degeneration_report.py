# When do replies degenerate in a long stream, and what are they made of? Reads the <video>_growth.json that demo.cli
# writes with --phase_timing True (decode rows carry reply, raw_reply, v_tokens, interval_tokens, ended_with_eos).
# A reply counts as degenerate if it hits the reply cap without an end-of-reply token, contains <v> frame placeholders,
# is empty, or contains chat-template syntax (']' closing a frame stream, 'Assistant'): a normal reply is a short step name.
# Usage: python -m profiling_scripts.degeneration_report <growth.json> [<growth.json> ...] [--window 5]
import argparse, json

def is_degenerate(e):
    text = e.get('reply', '')
    return ((not e.get('ended_with_eos', True)) or e.get('v_tokens', 0) > 0 or not text.strip()
            or ']' in text or 'Assistant' in text)

def report(path, window):
    rows = [e for e in json.load(open(path)) if e['phase'] == 'decode']
    if not rows or 'ended_with_eos' not in rows[0]:
        print(f'{path}: no reply details (run with the updated demo.cli)'); return
    bad = [e for e in rows if is_degenerate(e)]
    print('=' * 100); print(path)
    print(f'{len(rows)} replies, {len(bad)} degenerate, {sum(e["units"] for e in rows):,} reply tokens, '
          f'{sum(e["seconds"] for e in rows):,.0f} s decoding')
    if bad:
        f, o = bad[0], sustained_onset(rows)
        print(f'first degenerate reply: video {f["video_time"] / 60:.1f} min, KV cache {f["kv_tokens"]:,} tokens')
        if o:
            print(f'sustained breakdown from: video {o["video_time"] / 60:.1f} min, KV cache {o["kv_tokens"]:,} tokens')
        tok = sum(e['units'] for e in bad)
        print(f'in degenerate replies: {100 * sum(e["v_tokens"] for e in bad) / tok:.0f}% of tokens are <v> frame placeholders, '
              f'{100 * sum(e["interval_tokens"] for e in bad) / tok:.0f}% are frame-separator commas')
    else:
        print('no degenerate replies')
    print('%10s %8s %10s %9s %13s %12s' % ('minutes', 'replies', 'degenerate', 'avg_tok', 'kv_at_end', 'decode_s'))
    end = max(e['video_time'] for e in rows)
    for lo in range(0, int(end // 60) + 1, window):
        w = [e for e in rows if lo * 60 <= e['video_time'] < (lo + window) * 60]
        if w:
            print('%10s %8d %10d %9.1f %13s %12.0f' % (f'{lo}-{lo + window}', len(w), sum(map(is_degenerate, w)),
                  sum(e['units'] for e in w) / len(w), f'{max(e["kv_tokens"] for e in w):,}', sum(e['seconds'] for e in w)))
    print('examples:')
    good = [e for e in rows if not is_degenerate(e)]
    shown = good[:3] + bad[:2] + [e for e in bad[-2:] if e not in bad[:2]]  # no repeats when there are few
    for e in shown:
        tag = 'ok ' if not is_degenerate(e) else 'BAD'
        print(f'  {tag} {e["video_time"] / 60:5.1f} min | {e["units"]:3d} tok | v={e.get("v_tokens", 0):3d} | '
              f'{e["reply"][:60]!r} | raw {e["raw_reply"][:70]!r}')

def sustained_onset(decode_rows, lookahead=4, min_bad=3):
    """First degenerate reply followed by at least min_bad degenerate replies among the next lookahead (or all remaining):
    the point where replies break down for good, as opposed to a one-off glitch. Returns the row, or None."""
    for i, e in enumerate(decode_rows):
        if is_degenerate(e):
            nxt = decode_rows[i + 1:i + 1 + lookahead]
            if sum(map(is_degenerate, nxt)) >= min(min_bad, len(nxt)):
                return e
    return None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('growth_json', nargs='+')
    ap.add_argument('--window', type=int, default=5, help='minutes per row')
    a = ap.parse_args()
    for p in a.growth_json:
        report(p, a.window)
