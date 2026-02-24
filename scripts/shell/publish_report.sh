#!/usr/bin/env bash
# Publish the latest PCR benchmark report to the gh-pages branch.
#
# The HTML report is fully self-contained (all images embedded as base64),
# so gh-pages only ever holds one file: index.html.
#
# Usage:
#   scripts/shell/publish_report.sh                  # publish latest run
#   scripts/shell/publish_report.sh --results-dir /path/to/run  # specific run
#   scripts/shell/publish_report.sh --open           # also open in browser after
#   scripts/shell/publish_report.sh --local          # serve locally instead of pushing
#   scripts/shell/publish_report.sh --port 8080      # set local server port (default 8080)
#
# First-time setup:
#   git remote add github git@github.com:<owner>/<repo>.git
#   scripts/shell/publish_report.sh   # pushes to gh-pages; enable in Settings → Pages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Parse flags ────────────────────────────────────────────────────────────────
RESULTS_DIR=""
OPEN_AFTER=false
LOCAL_ONLY=false
PORT=8080

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --open)        OPEN_AFTER=true; shift ;;
        --local)       LOCAL_ONLY=true; shift ;;
        --port)        PORT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; echo "Usage: $0 [--results-dir DIR] [--open] [--local] [--port N]"; exit 1 ;;
    esac
done

# ── Find report ────────────────────────────────────────────────────────────────
if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="$WORKSPACE/benchmark_results/latest"
fi

REPORT="$RESULTS_DIR/benchmark_report.html"

if [ ! -f "$REPORT" ]; then
    echo "No report found at $REPORT"
    echo ""
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "Run the benchmark suite first:"
        echo "  scripts/shell/run_benchmarks.sh"
    else
        echo "Regenerate the report from existing results:"
        echo "  PYTHONPATH=$WORKSPACE/python python3 $WORKSPACE/scripts/benchmarks/generate_report.py \\"
        echo "    --results-dir $RESULTS_DIR"
    fi
    exit 1
fi

REPORT_SIZE=$(du -h "$REPORT" | cut -f1)
REPORT_DATE=$(date -r "$REPORT" '+%Y-%m-%d' 2>/dev/null || date -u '+%Y-%m-%d')

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              PCR Benchmark Report Publisher                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Report : $REPORT"
echo "  Size   : $REPORT_SIZE (standalone HTML, images embedded)"
echo "  Date   : $REPORT_DATE"
echo ""

# ── Local serve mode ───────────────────────────────────────────────────────────
if $LOCAL_ONLY; then
    REPORT_DIR="$(dirname "$REPORT")"
    echo "Serving locally at http://localhost:$PORT/"
    echo "Press Ctrl-C to stop."
    echo ""
    python3 -m http.server "$PORT" --directory "$REPORT_DIR"
    exit 0
fi

# ── Git plumbing publish — no working tree changes ─────────────────────────────
cd "$WORKSPACE"

echo "Building gh-pages branch..."

# 1. Store the HTML as a git blob
BLOB=$(git hash-object -w "$REPORT")

# 2. Create a tree: one file — index.html
TREE=$(printf '100644 blob %s\tindex.html\n' "$BLOB" | git mktree)

# 3. Chain to existing gh-pages history if it exists
PARENT_FLAG=""
if git rev-parse --verify refs/heads/gh-pages >/dev/null 2>&1; then
    PARENT_FLAG="-p $(git rev-parse refs/heads/gh-pages)"
fi

# 4. Create the commit (no working tree modified)
COMMIT=$(
    GIT_AUTHOR_NAME="PCR Benchmark Bot" \
    GIT_AUTHOR_EMAIL="benchmarks@pcr" \
    GIT_COMMITTER_NAME="PCR Benchmark Bot" \
    GIT_COMMITTER_EMAIL="benchmarks@pcr" \
    git commit-tree "$TREE" $PARENT_FLAG \
        -m "publish: benchmark report $REPORT_DATE"
)

# 5. Advance the branch ref
git update-ref refs/heads/gh-pages "$COMMIT"
echo "  gh-pages → $COMMIT"

# ── Push to GitHub remote ──────────────────────────────────────────────────────
# Finds any remote whose URL contains github.com; skips local/SSH-private remotes.
GITHUB_REMOTE=""
for remote in $(git remote); do
    url=$(git remote get-url "$remote" 2>/dev/null || true)
    if echo "$url" | grep -qE 'github\.com'; then
        GITHUB_REMOTE="$remote"
        break
    fi
done

if [ -n "$GITHUB_REMOTE" ]; then
    echo "  Pushing to $GITHUB_REMOTE..."
    git push "$GITHUB_REMOTE" gh-pages --force

    # Derive the Pages URL from the remote URL
    RAW=$(git remote get-url "$GITHUB_REMOTE")
    # Strip protocol/host, handle both HTTPS and SSH formats
    SLUG=$(echo "$RAW" \
        | sed 's|.*github\.com[:/]||' \
        | sed 's|\.git$||')
    OWNER=$(echo "$SLUG" | cut -d/ -f1)
    REPO=$(echo "$SLUG" | cut -d/ -f2)
    PAGES_URL="https://${OWNER}.github.io/${REPO}/"

    echo ""
    echo "  Deployed!"
    echo ""
    echo "  URL: $PAGES_URL"
    echo ""
    echo "  If this is your first deploy, enable GitHub Pages:"
    echo "    github.com/$SLUG → Settings → Pages → Source: gh-pages"
    echo ""

    if $OPEN_AFTER; then
        xdg-open "$PAGES_URL" 2>/dev/null || open "$PAGES_URL" 2>/dev/null || true
    fi
else
    echo ""
    echo "  No GitHub remote found — branch updated locally only."
    echo ""
    echo "  To push, add a GitHub remote and re-run:"
    echo "    git remote add github git@github.com:<owner>/<repo>.git"
    echo "    scripts/shell/publish_report.sh"
    echo ""
    echo "  Or serve the report locally right now:"
    echo "    scripts/shell/publish_report.sh --local"
    echo ""
    echo "  Or open the HTML file directly (self-contained, no server needed):"
    echo "    open $REPORT"

    if $OPEN_AFTER; then
        xdg-open "$REPORT" 2>/dev/null || open "$REPORT" 2>/dev/null || true
    fi
fi
