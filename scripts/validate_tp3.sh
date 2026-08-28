#!/usr/bin/env bash
# Validate a TP=3 GLM-5.3-Flash deployment.
#
# "It started" is NOT evidence of correctness. A wrong head shard, a non-zero
# pad slab, or a mis-loaded expert column produces fluent, plausible text with
# no error anywhere -- that is the entire hazard the TP=3 patch exists to
# avoid. So this checks reasoning that subtly-broken attention would fail,
# not just that tokens come out.
#
# GLM specifics vs the DSv4 edition of this script:
#   * every probe sends chat_template_kwargs {"enable_thinking": false} --
#     GLM-5.3 is a reasoning model, and with thinking on the answer can land
#     in reasoning_content while content stays empty, which would fail every
#     probe for a reason that has nothing to do with sharding.
#   * a sequential-state probe (KDA linear-attention layers carry per-head
#     recurrent state; 34 of 45 layers are KDA).
#   * a multimodal probe (solid-color PNG) -- the vision tower is handled
#     differently from the text stack under TP=3 (replicated), and only a
#     real image proves that path.
#
# Usage: validate_tp3.sh [host:port]        (default 127.0.0.1:8888)
#
set -uo pipefail

EP="${1:-127.0.0.1:8888}"
BASE="http://$EP"
pass=0; fail=0

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
skip() { printf '  \033[33mSKIP\033[0m %s\n' "$1"; }
info() { printf '  ---- %s\n' "$1"; }

echo "=== 1. server up + model identity ==="
models="$(curl -s -m 20 "$BASE/v1/models" 2>/dev/null)"
if [[ -z "$models" ]]; then
  bad "no response from $BASE/v1/models"
  echo; echo "Server is not answering. Nothing else can be validated."; exit 1
fi
mid="$(printf '%s' "$models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
[[ -n "$mid" ]] && ok "serving: $mid" || bad "could not parse model id"

ask() {  # $1=prompt  $2=max_tokens
  curl -s -m 240 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],
                  "max_tokens":int(sys.argv[3]),"temperature":0,
                  "chat_template_kwargs":{"enable_thinking":False}}))' "$mid" "$1" "${2:-64}")" \
  | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin)
    m=d["choices"][0]["message"]
    c=(m.get("content") or "").strip()
    # if the template kwarg was ignored and everything went to the thinking
    # channel, fall back to reasoning_content rather than failing on plumbing
    if not c:
        c=(m.get("reasoning_content") or "").strip()
    print(c)
except Exception as e:
    print(f"__ERROR__ {e}")' 2>/dev/null
}

echo
echo "=== 2. deterministic factual recall ==="
# Degenerate attention usually survives token-level fluency but loses precise
# recall first. These have exactly one right answer.
for probe in "What is the capital of Japan?|Tokyo" \
             "What is 17 multiplied by 23? Reply with only the number.|391" \
             "Complete exactly: The quick brown fox jumps over the lazy|dog"; do
  q="${probe%%|*}"; want="${probe##*|}"
  a="$(ask "$q" 48)"
  if [[ "$a" == __ERROR__* ]]; then bad "request failed: $q ($a)"
  elif grep -qi -- "$want" <<<"$a"; then ok "$q -> $(head -c 60 <<<"$a")"
  else bad "$q -> expected '$want', got: $(head -c 90 <<<"$a")"; fi
done

echo
echo "=== 3. multi-step reasoning (fails first under a bad shard) ==="
a="$(ask 'A shelf holds 3 red books and 5 blue books. I remove 2 blue books, then add 4 red books. How many red and how many blue remain? Answer in the form: red=N blue=N' 96)"
if [[ "$a" == __ERROR__* ]]; then bad "reasoning request failed ($a)"
elif grep -qE 'red=7' <<<"$a" && grep -qE 'blue=3' <<<"$a"; then ok "arithmetic reasoning: $(head -c 70 <<<"$a")"
else bad "expected red=7 blue=3, got: $(head -c 120 <<<"$a")"; fi

echo
echo "=== 4. sequential state (34 of 45 layers are KDA linear attention) ==="
# KDA layers carry fixed-size per-head recurrent state instead of paged KV.
# A mis-sharded or garbage-padded KDA head corrupts exactly this kind of
# running-state task while leaving one-shot recall plausible.
a="$(ask 'Count from 1 to 12 as digits separated by commas, nothing else.' 64)"
if [[ "$a" == __ERROR__* ]]; then bad "sequence request failed ($a)"
elif grep -q '1, *2, *3, *4, *5, *6, *7, *8, *9, *10, *11, *12' <<<"$a"; then ok "sequential count intact"
else bad "sequence corrupted -> $(head -c 100 <<<"$a")"; fi

echo
echo "=== 5. long-range coherence (attention over distance) ==="
needle="The passphrase is CRIMSON-MERIDIAN-42."
filler="$(python3 -c 'print(" ".join(["The weather report for that day was unremarkable."]*120))')"
a="$(ask "$needle $filler What exactly is the passphrase? Reply with only the passphrase." 32)"
if [[ "$a" == __ERROR__* ]]; then bad "needle test failed ($a)"
elif grep -qi 'CRIMSON-MERIDIAN-42' <<<"$a"; then ok "recalled the needle across ~1.5k tokens"
else bad "lost the needle -> $(head -c 90 <<<"$a")"; fi

echo
echo "=== 6. degeneration check (repetition / gibberish) ==="
a="$(ask 'Write two sentences about the ocean.' 80)"
if [[ "$a" == __ERROR__* ]]; then bad "generation failed ($a)"
else
  words=$(wc -w <<<"$a"); uniq=$(tr ' ' '\n' <<<"$a" | sort -u | wc -l)
  ratio=$(python3 -c "print(round($uniq/max($words,1),2))")
  info "words=$words unique=$uniq ratio=$ratio"
  python3 -c "import sys; sys.exit(0 if $ratio >= 0.5 else 1)" \
    && ok "no degeneration (unique-word ratio $ratio)" \
    || bad "possible degeneration, ratio $ratio: $(head -c 100 <<<"$a")"
fi

echo
echo "=== 7. multimodal (vision tower path under TP=3) ==="
# An 8x8 solid red truecolor PNG as a data URL. The vision tower has 16 heads (16 % 3
# != 0) and is REPLICATED under this patch rather than sharded -- only a real
# image through the API proves that path. Best-effort: a text-only serving
# profile skips rather than fails.
png_b64="iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP4z8CAFWEXHbQSACj/P8Fu7N9hAAAAAElFTkSuQmCC"
mm_resp="$(curl -s -m 240 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,"+sys.argv[2]}},
    {"type":"text","text":"What single color dominates this image? Answer with one word."}]}],
  "max_tokens":48,"temperature":0,
  "chat_template_kwargs":{"enable_thinking":False}}))' "$mid" "$png_b64")" 2>/dev/null)"
mm_a="$(printf '%s' "$mm_resp" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin)
    if "error" in d: print("__API_ERROR__ "+str(d["error"])[:120]); raise SystemExit
    m=d["choices"][0]["message"]
    print(((m.get("content") or m.get("reasoning_content") or "")).strip())
except SystemExit: pass
except Exception as e: print(f"__ERROR__ {e}")' 2>/dev/null)"
if [[ "$mm_a" == __API_ERROR__* ]]; then
  skip "multimodal not accepted by this serving profile ($(head -c 80 <<<"$mm_a"))"
elif [[ "$mm_a" == __ERROR__* || -z "$mm_a" ]]; then bad "multimodal request failed ($mm_a)"
elif grep -qi 'red' <<<"$mm_a"; then ok "vision path: solid-red image -> $(head -c 40 <<<"$mm_a")"
else bad "vision path answered wrong color -> $(head -c 90 <<<"$mm_a")"; fi

echo
echo "==============================================="
printf '  %d passed, %d failed\n' "$pass" "$fail"
if (( fail )); then
  echo "  ⚠️  DO NOT TRUST THIS DEPLOYMENT until these are explained."
  echo "     A wrong head shard is silent -- fluent output is not proof."
  exit 1
fi
echo "  All checks passed."
