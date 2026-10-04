#!/usr/bin/env bash
# Sync the fork with upstream vLLM.
#
# The fork keeps `main` as an exact mirror of vllm-project/vllm main and
# carries its own work as a linear stack of commits on `gfx1151`. Syncing
# rebases that stack onto the newest upstream main. Nothing is pushed
# without --push: a clean rebase can still break things semantically, so
# run the tests (and a real serve) on the rebased branch first.
#
# If this checkout is an editable install, rebasing changes the code that
# running vLLM processes load lazily; stop servers first.
#
# Usage: tools/sync-upstream.sh [--check | --push]
#   --check  report how far behind the fork is; change nothing
#   --push   rebase if needed, then fast-forward origin's main to upstream
#            and force-push the work branch (with lease)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/vllm-project/vllm.git}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
MIRROR_BRANCH="${MIRROR_BRANCH:-main}"
WORK_BRANCH="${WORK_BRANCH:-gfx1151}"
PYTHON="${PYTHON:-python}"
MODE="${1:-}"

die() { echo "error: $*" >&2; exit 1; }

case "$MODE" in
  "" | --check | --push) ;;
  *) die "usage: $0 [--check | --push]" ;;
esac

if [[ -d "$(git rev-parse --git-path rebase-merge)" ||
      -d "$(git rev-parse --git-path rebase-apply)" ]]; then
  die "a rebase is already in progress; finish it or run: git rebase --abort"
fi
[[ -z "$(git status --porcelain --untracked-files=no)" ]] ||
  die "working tree has uncommitted changes; commit or stash first"
[[ "$(git branch --show-current)" == "$WORK_BRANCH" ]] ||
  die "checkout $WORK_BRANCH first"

echo "fetching $UPSTREAM_URL $UPSTREAM_BRANCH ..."
git fetch -q "$UPSTREAM_URL" "$UPSTREAM_BRANCH"
UPSTREAM_HEAD=$(git rev-parse FETCH_HEAD)
BASE=$(git merge-base HEAD "$UPSTREAM_HEAD")
OURS=$(git rev-list --count "$BASE..HEAD")
BEHIND=$(git rev-list --count "$BASE..$UPSTREAM_HEAD")
echo "state: $WORK_BRANCH carries $OURS commits; upstream has $BEHIND new"

if [[ "$MODE" == "--check" ]]; then
  if (( BEHIND == 0 )); then
    echo "already up to date"
  elif git merge-tree --write-tree HEAD "$UPSTREAM_HEAD" >/dev/null; then
    echo "no conflicts expected"
  else
    echo "conflicts expected in:"
    git merge-tree --write-tree --name-only --no-messages HEAD "$UPSTREAM_HEAD" |
      tail -n +2 | sed 's/^/  /'
  fi
  exit 0
fi

if (( BEHIND > 0 )); then
  OLD_HEAD=$(git rev-parse HEAD)
  echo "rebasing onto $(git rev-parse --short "$UPSTREAM_HEAD") (old head: $OLD_HEAD)"
  if ! git rebase "$UPSTREAM_HEAD"; then
    cat >&2 <<EOF
rebase stopped on a conflict. Resolve the files, then:
  git add <files> && git rebase --continue
or restore the branch as it was with:
  git rebase --abort
EOF
    exit 1
  fi
  echo "undo with: git reset --hard $OLD_HEAD"

  # The fork's KV-cache pool API must still resolve after the rebase.
  "$PYTHON" -c "
from vllm.v1.core.block_pool import BlockPoolCollection
from vllm.v1.kv_cache_interface import KVCachePoolSpec
" || die "fork KV-cache API no longer imports; fix before pushing"
fi

if [[ "$MODE" != "--push" ]]; then
  echo "done locally. Run the tests, then publish with: $0 --push"
  exit 0
fi

REMOTE_WORK=$(git ls-remote origin "refs/heads/$WORK_BRANCH" | cut -f1)
git push origin "$UPSTREAM_HEAD:refs/heads/$MIRROR_BRANCH"
git push --force-with-lease="$WORK_BRANCH:$REMOTE_WORK" origin "$WORK_BRANCH"
echo "pushed: $MIRROR_BRANCH = upstream, $WORK_BRANCH = $(git rev-parse --short HEAD)"
