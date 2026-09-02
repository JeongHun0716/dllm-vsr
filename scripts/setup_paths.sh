#!/usr/bin/env bash
# One-time setup: fill in your dataset video roots.
#
# manifest/lrs3/*.tsv has `{LRS3_ROOT}` as line 0 (the directory under which
# the per-utterance mp4 paths live). After running this script, that placeholder
# is replaced by your absolute LRS3 mouth-ROI video root.
# manifest/wildvsr/wildvsr.tsv has `{WILDVSR_ROOT}` likewise (optional 2nd arg).
#
# Usage:
#   bash scripts/setup_paths.sh /abs/path/to/lrs3_mouth_rois_root [/abs/path/to/WildVSR]
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <LRS3_VIDEO_ROOT> [WILDVSR_ROOT]"
    exit 1
fi
LRS3_ROOT="$1"
WILDVSR_ROOT="${2:-}"

patch_manifests() {
    local root="$1" placeholder="$2"; shift 2
    if [[ ! -d "${root}" ]]; then
        echo "error: ${root} does not exist or is not a directory" >&2
        exit 2
    fi
    # Escape for sed (paths may contain /).
    local escaped
    escaped=$(printf '%s\n' "${root}" | sed 's:[\/&]:\\&:g')
    for f in "$@"; do
        if head -1 "$f" | grep -qF "${placeholder}"; then
            sed -i "1s|.*|${escaped}|" "$f"
            echo "patched: $f  ->  line 0 = ${root}"
        else
            echo "skip:    $f  (line 0 already set: $(head -1 "$f"))"
        fi
    done
}

patch_manifests "${LRS3_ROOT}" '{LRS3_ROOT}' manifest/lrs3/*.tsv
if [[ -n "${WILDVSR_ROOT}" ]]; then
    patch_manifests "${WILDVSR_ROOT}" '{WILDVSR_ROOT}' manifest/wildvsr/*.tsv
fi

echo
echo "Done. Verify a sample:"
echo "  head -2 manifest/lrs3/test.tsv"
