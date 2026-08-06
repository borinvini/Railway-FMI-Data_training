# Vendored third-party tools

## csc_cert.py

CSC's SSH certificate helper, used by `hpc/roihu-auth.sh` to sign the
short-lived certificate Roihu requires.

- Upstream: https://github.com/CSCfi/certificate-helper-tool
- Pinned version: **v1.0.0** (released 2026-06-28)
- License: MIT, CSC - IT Center for Science (see `LICENSE`)
- Dependencies: Python 3.8+ only, no third-party packages

Vendored verbatim. **Do not edit.** To update, re-download at the new tag and
bump the version above:

```bash
curl -sSf -o hpc/vendor/csc_cert.py \
  "https://raw.githubusercontent.com/CSCfi/certificate-helper-tool/<tag>/csc_cert.py"
```

After updating, confirm `-a` still accepts `none` and `-p/--no-ppk` still
exists — `hpc/roihu-auth.sh` passes both.
