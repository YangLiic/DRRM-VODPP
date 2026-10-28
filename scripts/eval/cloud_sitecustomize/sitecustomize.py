import os
import site

# Loaded automatically by CPython's site module when this directory is on
# PYTHONPATH. Needed because Volcano cannot use robotwin .venv/bin/python
# (it is a host-bound symlink to /root/miniconda3/...), so we run image
# python3.10 against the venv site-packages. Plain PYTHONPATH=site-packages
# does NOT process .pth / editable finders (curobo, vggt); addsitedir does.
_site = os.environ.get("ROBOTWIN_VENV_SITE")
if _site:
    site.addsitedir(_site)
