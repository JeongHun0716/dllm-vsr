#!/usr/bin/env bash
# One-time setup: fill in your LRS3 raw video root.
#
# manifest/433h/*.tsv has `{LRS3_ROOT}` as line 0 (the directory under which
# the per-utterance mp4 paths live). After running this script, that placeholder
# is replaced by your absolute LRS3 video root.
#
# Usage:
#   bash scripts/setup_paths.sh /abs/path/to/lrs3_videos
#
# `train_excl_val.tsv` is also handled if present.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <LRS3_VIDEO_ROOT>"
    exit 1
fi
LRS3_ROOT="$1"

if [[ ! -d "${LRS3_ROOT}" ]]; then
    echo "error: ${LRS3_ROOT} does not exist or is not a directory" >&2
    exit 2
fi

# Escape for sed (paths may contain /).
ESCAPED=$(printf '%s\n' "${LRS3_ROOT}" | sed 's:[\/&]:\\&:g')

for f in manifest/433h/*.tsv; do
    if head -1 "$f" | grep -qF '{LRS3_ROOT}'; then
        sed -i "1s|.*|${ESCAPED}|" "$f"
        echo "patched: $f  ->  line 0 = ${LRS3_ROOT}"
    else
        echo "skip:    $f  (line 0 already set: $(head -1 "$f"))"
    fi
done

echo
echo "Done. Verify a sample:"
echo "  head -2 manifest/433h/test.tsv"
