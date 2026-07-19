#!/usr/bin/env bash
set -euo pipefail

readonly ATALANTA_REPOSITORY="https://github.com/hsluoyz/Atalanta.git"
readonly ATALANTA_COMMIT="a8e07fe4af80c55b0d4ca77e382731b03ad731dc"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/tools/atalanta"

if [[ -e "${source_dir}" && ! -d "${source_dir}/.git" ]]; then
    echo "error: ${source_dir} exists but is not an Atalanta Git checkout" >&2
    exit 1
fi

if [[ ! -d "${source_dir}/.git" ]]; then
    mkdir -p "$(dirname "${source_dir}")"
    git clone "${ATALANTA_REPOSITORY}" "${source_dir}"
fi

git -C "${source_dir}" fetch --depth 1 origin "${ATALANTA_COMMIT}"
git -C "${source_dir}" checkout --detach "${ATALANTA_COMMIT}"
make -C "${source_dir}"

echo "Atalanta installed at ${source_dir}/atalanta"
echo "Run it through ${repo_root}/bin/atalanta"
