#!/usr/bin/env bash
# Shared Volcano / cloud headless env for Sapien + robotwin packages.
# Source from *_cloud.sh only — do not use on local workstations.
#
# Root cause of past cloud failures:
#   robotwin .venv/bin/python -> /root/miniconda3/envs/robotics_manipulation/...
#   which does not exist on workers. Falling back to /usr/bin/python3 with a
#   bare site-packages PYTHONPATH misses editable installs (curobo / vggt).
#
# Fix:
#   Use image python3.10 + ROBOTWIN_VENV_SITE + cloud_sitecustomize/sitecustomize.py
#   so site.addsitedir() processes all .pth / editable finders in every process
#   (including multiprocessing spawn children).

_CLOUD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CLOUD_PROJECT_ROOT="$(cd "$_CLOUD_SCRIPT_DIR/../.." && pwd)"
_CLOUD_ROBOTWIN="$_CLOUD_PROJECT_ROOT/simulation/robotwin2.0"
_CLOUD_VENV="$_CLOUD_ROBOTWIN/.venv"
_CLOUD_SITE="$_CLOUD_VENV/lib/python3.10/site-packages"
_CLOUD_SITECUSTOMIZE="$_CLOUD_SCRIPT_DIR/cloud_sitecustomize"

export SAPIEN_EGL_NO_X11="${SAPIEN_EGL_NO_X11:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export KRAKEN_EGL_DEVICE="${KRAKEN_EGL_DEVICE:-${GPU_ID:-0}}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if [ ! -d "$_CLOUD_SITE" ]; then
  echo "[cloud-env] ERROR: missing venv site-packages: $_CLOUD_SITE" >&2
  return 1 2>/dev/null || exit 1
fi
if [ ! -f "$_CLOUD_SITECUSTOMIZE/sitecustomize.py" ]; then
  echo "[cloud-env] ERROR: missing $_CLOUD_SITECUSTOMIZE/sitecustomize.py" >&2
  return 1 2>/dev/null || exit 1
fi

# Prefer a working venv interpreter when the symlink resolves on this machine;
# otherwise use image python3.10 + sitecustomize addsitedir.
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$_CLOUD_VENV/bin/python" ] \
    && "$_CLOUD_VENV/bin/python" -c "import torch, curobo, vggt" >/dev/null 2>&1; then
    export PYTHON="$_CLOUD_VENV/bin/python"
  else
    _py=""
    for _cand in /usr/bin/python3.10 python3.10; do
      if [ -x "$_cand" ]; then
        _py="$_cand"
        break
      fi
      if command -v "$_cand" >/dev/null 2>&1; then
        _py="$(command -v "$_cand")"
        break
      fi
    done
    if [ -z "$_py" ]; then
      echo "[cloud-env] ERROR: no python3.10 on image" >&2
      return 1 2>/dev/null || exit 1
    fi
    export PYTHON="$_py"
    export ROBOTWIN_VENV_SITE="$_CLOUD_SITE"
    export VIRTUAL_ENV="$_CLOUD_VENV"
    export PATH="${_CLOUD_VENV}/bin:${PATH}"
    # Prepend sitecustomize dir so site.py imports our sitecustomize.py.
    # Do NOT put site-packages itself on PYTHONPATH here — addsitedir must own it.
    export PYTHONPATH="${_CLOUD_SITECUSTOMIZE}${PYTHONPATH:+:${PYTHONPATH}}"
    unset _cand _py
  fi
fi

# Always keep sitecustomize available if we are on the fallback path.
if [ -n "${ROBOTWIN_VENV_SITE:-}" ]; then
  case ":${PYTHONPATH:-}:" in
    *":${_CLOUD_SITECUSTOMIZE}:"*) ;;
    *) export PYTHONPATH="${_CLOUD_SITECUSTOMIZE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
fi

if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] \
  && [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  mkdir -p /tmp/vulkan_icd
  _vk_lib=""
  for _cand in \
    /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 \
    /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so \
    $(ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.* 2>/dev/null | grep -v '\.so\.0$' | tail -n 1)
  do
    if [ -n "${_cand}" ] && [ -e "${_cand}" ]; then
      _vk_lib="${_cand}"
      break
    fi
  done
  if [ -n "${_vk_lib}" ]; then
    cat > /tmp/vulkan_icd/nvidia_egl.json <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "${_vk_lib}",
        "api_version" : "1.3.242"
    }
}
EOF
    export VK_ICD_FILENAMES=/tmp/vulkan_icd/nvidia_egl.json
  elif [ -f /tmp/vulkan_icd/nvidia_egl.json ]; then
    export VK_ICD_FILENAMES=/tmp/vulkan_icd/nvidia_egl.json
  fi
  unset _vk_lib _cand
fi

# One-shot preflight: fail fast with a clear error instead of hanging after Render Well.
echo "[cloud-env] preflight PYTHON=$PYTHON ROBOTWIN_VENV_SITE=${ROBOTWIN_VENV_SITE:-unset}"
if ! "$PYTHON" - <<'PY'
import importlib
required = ["torch", "curobo", "vggt", "sapien", "cv2", "yaml", "einops", "transformers"]
failed = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{name}: {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit("missing imports:\n  " + "\n  ".join(failed))
print("preflight imports ok", flush=True)
PY
then
  echo "[cloud-env] ERROR: preflight failed" >&2
  return 1 2>/dev/null || exit 1
fi

unset _CLOUD_SCRIPT_DIR _CLOUD_PROJECT_ROOT _CLOUD_ROBOTWIN _CLOUD_VENV _CLOUD_SITE _CLOUD_SITECUSTOMIZE
