#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLASS_DIR="${1:-$ROOT/.runtime/class_iv}"
BASE_COMMIT="ac627d54e9ce196a08878d1ba33999819925d19c"

mkdir -p "$(dirname "$CLASS_DIR")"

if [[ -e "$CLASS_DIR" ]]; then
  printf 'Refusing to overwrite existing path: %s\n' "$CLASS_DIR" >&2
  exit 2
fi

git clone https://github.com/kaeonikc/class_iv.git "$CLASS_DIR"
git -C "$CLASS_DIR" checkout --detach "$BASE_COMMIT"

git -C "$CLASS_DIR" apply --check "$ROOT/patches/00_base_fix_ac627d5.diff"
git -C "$CLASS_DIR" apply "$ROOT/patches/00_base_fix_ac627d5.diff"
git -C "$CLASS_DIR" apply --check "$ROOT/patches/01_test53_CLASS_canonical.diff"
git -C "$CLASS_DIR" apply "$ROOT/patches/01_test53_CLASS_canonical.diff"

grep -R -q "f_dyn_test53" "$CLASS_DIR/include" "$CLASS_DIR/source"
cmp "$ROOT/model/test53_shape.h" "$CLASS_DIR/source/test53_shape.h"

make -C "$CLASS_DIR" clean >/dev/null 2>&1 || true
if ! make -C "$CLASS_DIR" -j"$(nproc)" class OPTFLAG="-O2 -fcommon"; then
  test -n "$(find "$CLASS_DIR/build" -maxdepth 1 -name '*.o' -print -quit)"
  gcc -O2 -fcommon -fopenmp -o "$CLASS_DIR/class" \
    "$CLASS_DIR"/build/*.o -lgsl -lgslcblas -lm
fi

test -x "$CLASS_DIR/class"
strings "$CLASS_DIR/class" | grep -qF "f_dyn_test53"
printf 'CLASS Test 53 built at %s\n' "$CLASS_DIR/class"
