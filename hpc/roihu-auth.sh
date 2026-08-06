#!/bin/bash
# Sign a fresh Roihu SSH certificate, load the agent, report the expiry.
#
# Roihu rejects both passwords and bare public keys. It needs a CSC-signed
# certificate valid 24 hours, on top of the never-expiring public key
# registered in MyCSC. This script is the whole daily ritual:
#
#   hpc/roihu-auth.sh        # sign if needed
#   hpc/roihu-auth.sh -r     # force re-sign even if still valid
#
# Safe to run repeatedly: csc_cert.py exits 0 without opening a browser when
# the current certificate is still valid.
#
# Signing cannot be made unattended. MyCSC authenticates through a device-code
# flow — browser login plus a 6-digit code — and CSC publishes no API token.
# One command in ~20 seconds is the floor, not a shortcoming of this script.
#
# Environment overrides:
#   CSC_USER              CSC username         (default vpozzobo)
#   SSH_KEY               private key path     (default ~/.ssh/id_csc)
#   ROIHU_AUTH_NO_AGENT   set to skip all agent handling — sign only, and accept
#                         retyping the passphrase on every connection
set -euo pipefail

CSC_USER="${CSC_USER:-vpozzobo}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_csc}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_TOOL="${REPO_ROOT}/hpc/vendor/csc_cert.py"
AGENT_ENV="${HOME}/.ssh/agent.env"
CERT="${SSH_KEY}-cert.pub"

# Pin the ssh tools to Git Bash's own OpenSSH. On Windows, bare `ssh-add` /
# `ssh-agent` / `ssh-keygen` resolve off PATH, and if
# C:\Windows\System32\OpenSSH precedes /usr/bin there, ssh-add would talk to
# the Windows named-pipe agent while Git's ssh uses the unix socket agent we
# set up below — a silent split that leaves every later `ssh roihu` demanding
# the passphrase despite this script printing "Ready.".
SSH_ADD="/usr/bin/ssh-add"
SSH_AGENT="/usr/bin/ssh-agent"
SSH_KEYGEN="/usr/bin/ssh-keygen"
for tool_var in SSH_ADD SSH_AGENT SSH_KEYGEN; do
    tool_path="${!tool_var}"
    if [ ! -x "${tool_path}" ]; then
        echo "ERROR: ${tool_path} not found or not executable." >&2
        echo "       This script requires Git Bash's own OpenSSH, not Windows'" >&2
        echo "       C:\\Windows\\System32\\OpenSSH build, to keep ssh-add and ssh" >&2
        echo "       talking to the same agent." >&2
        exit 1
    fi
done

REFRESH=()
while getopts "r" opt; do
    case "${opt}" in
        r) REFRESH=(-r) ;;
        *) echo "Usage: hpc/roihu-auth.sh [-r]" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
if [ "$#" -gt 0 ]; then
    echo "Usage: hpc/roihu-auth.sh [-r]" >&2
    echo "ERROR: unexpected argument(s): $*" >&2
    exit 2
fi

# --- Preflight -------------------------------------------------------------
if [ ! -f "${SSH_KEY}" ]; then
    echo "ERROR: no private key at ${SSH_KEY}" >&2
    echo "       Create one, then register the .pub in MyCSC (Profile -> SSH PUBLIC KEYS):" >&2
    echo "         ssh-keygen -t ed25519 -f ${SSH_KEY} -C \"${CSC_USER}@csc\"" >&2
    exit 1
fi

if [ ! -f "${SSH_KEY}.pub" ]; then
    echo "ERROR: no public key at ${SSH_KEY}.pub" >&2
    echo "       Regenerate it from the private key:" >&2
    echo "         ssh-keygen -y -f \"${SSH_KEY}\" > \"${SSH_KEY}.pub\"" >&2
    exit 1
fi

if [ ! -f "${CERT_TOOL}" ]; then
    echo "ERROR: hpc/vendor/csc_cert.py not found at ${CERT_TOOL}" >&2
    echo "       Re-vendor it (see hpc/vendor/README.md)." >&2
    exit 1
fi

# python3 does not exist in Git Bash here, and bare `python` can hit the
# Windows Store shim in some shells. Miniconda's interpreter is the known-good
# one, so fall back to it explicitly rather than failing obscurely inside the
# certificate tool.
PYTHON="python"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    PYTHON="/c/Users/${USERNAME:-$(whoami)}/miniconda3/python.exe"
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "ERROR: no usable Python. Expected 'python' on PATH, or miniconda at" >&2
    echo "       ${PYTHON}" >&2
    exit 1
fi

# --- 1. Reuse or start the Git Bash ssh-agent ------------------------------
# Windows has two unrelated agents: the ssh-agent service (named pipe, serves
# C:\Windows\System32\OpenSSH\ssh.exe) and whatever Git Bash's ssh talks to
# (unix socket, $SSH_AUTH_SOCK). This project is entirely Git Bash, so we use
# the socket agent and persist its env for other terminals.
#
# ssh-add -l exits 0 with keys loaded, 1 when the agent is up but empty, and 2
# when no agent is reachable. Only 2 means dead.
agent_alive() {
    [ -n "${SSH_AUTH_SOCK:-}" ] || return 1
    [ -S "${SSH_AUTH_SOCK}" ] || return 1
    local rc=0
    "${SSH_ADD}" -l >/dev/null 2>&1 || rc=$?
    [ "${rc}" -ne 2 ]
}

if [ -z "${ROIHU_AUTH_NO_AGENT:-}" ]; then
    if ! agent_alive && [ -f "${AGENT_ENV}" ]; then
        # shellcheck source=/dev/null
        . "${AGENT_ENV}" >/dev/null
    fi
    if ! agent_alive; then
        echo "Starting a new ssh-agent..."
        if (umask 077; "${SSH_AGENT}" -s > "${AGENT_ENV}"); then
            chmod 600 "${AGENT_ENV}"
            # shellcheck source=/dev/null
            . "${AGENT_ENV}" >/dev/null
        else
            rm -f "${AGENT_ENV}"
        fi
    fi
    if ! agent_alive; then
        echo "ERROR: could not start an ssh-agent. Refusing to continue agentless," >&2
        echo "       which would mean retyping the passphrase on every connection." >&2
        echo "       Set ROIHU_AUTH_NO_AGENT=1 to sign anyway without an agent." >&2
        exit 1
    fi
fi

# --- 2. Sign ---------------------------------------------------------------
# -a none: the tool's Windows default is 'both', which targets the named-pipe
#          agent Git Bash cannot reach. We do ssh-add ourselves, below.
# -p:      skip PuTTY .ppk creation, which needs WinSCP (not installed). NOT
#          implied by -a none — create_ppk defaults True (csc_cert.py:60,463).
"${PYTHON}" "${CERT_TOOL}" -u "${CSC_USER}" -a none -p "${REFRESH[@]}" "${SSH_KEY}.pub"

# --- 3. Load the key into the agent ----------------------------------------
# A wrong passphrase or Ctrl-C here must not abort the whole run: signing
# already succeeded above, so failing to load the agent is a degraded-but-
# usable outcome, not a reason to hide a good certificate's expiry.
if [ -z "${ROIHU_AUTH_NO_AGENT:-}" ]; then
    KEY_FP="$("${SSH_KEYGEN}" -lf "${SSH_KEY}.pub" 2>/dev/null | awk '{print $2}')" || KEY_FP=""
    if [ -z "${KEY_FP}" ] || ! "${SSH_ADD}" -l 2>/dev/null | grep -qF "${KEY_FP}"; then
        echo "Adding ${SSH_KEY} to the agent (passphrase needed once per boot)..."
        if ! "${SSH_ADD}" "${SSH_KEY}"; then
            echo "WARNING: key not loaded into the agent; the certificate is still valid." >&2
        fi
    fi
fi

# --- 4. Report ---------------------------------------------------------------
# Exit 0 from csc_cert.py is not proof a certificate landed, and a leftover
# expired certificate from a prior day must not be reported as fresh success.
if [ ! -f "${CERT}" ]; then
    echo "ERROR: signing reported success but no certificate was found at ${CERT}" >&2
    exit 1
fi

VALID_LINE="$("${SSH_KEYGEN}" -L -f "${CERT}" 2>/dev/null | grep "Valid:" | sed 's/^ *//')" || VALID_LINE=""
if [ -z "${VALID_LINE}" ]; then
    echo "ERROR: could not read/parse certificate validity from ${CERT}" >&2
    exit 1
fi

VALID_END="${VALID_LINE##* }"
VALID_END_EPOCH="$(date -d "${VALID_END/T/ }" +%s 2>/dev/null)" || VALID_END_EPOCH=""
if [ -z "${VALID_END_EPOCH}" ]; then
    echo "WARNING: could not check certificate expiry (unparseable timestamp)." >&2
    echo "${VALID_LINE}"
else
    NOW_EPOCH="$(date +%s)"
    if [ "${VALID_END_EPOCH}" -lt "${NOW_EPOCH}" ]; then
        echo "ERROR: certificate at ${CERT} is expired:" >&2
        echo "       ${VALID_LINE}" >&2
        exit 1
    fi
    echo "${VALID_LINE}"
fi
echo "Ready. Try: ssh roihu \"echo OK\""
