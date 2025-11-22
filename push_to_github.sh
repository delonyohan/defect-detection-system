    #!/usr/bin/env bash
    set -euo pipefail

    # USAGE:
    #   export GITHUB_TOKEN="ghp_..."
    #   ./push_to_github.sh <github-username>

    if [ -z "${GITHUB_TOKEN:-}" ]; then
      echo "ERROR: Please export GITHUB_TOKEN before running."
      exit 1
    fi

    if [ $# -lt 1 ]; then
      echo "Usage: $0 <github-username> [repo-name]"
      exit 1
    fi

    GITHUB_USER="$1"
    REPO_NAME="${2:-defect-detection-system}"
    PRIVATE=true

    echo "Creating repo ${GITHUB_USER}/${REPO_NAME}..."

    create_resp=$(curl -s -X POST "https://api.github.com/user/repos"       -H "Authorization: token ${GITHUB_TOKEN}"       -H "Accept: application/vnd.github.v3+json"       -d "{"name":"$REPO_NAME","private":$PRIVATE}")

    clone_url=$(echo "$create_resp" | python3 -c "import sys, json; r=json.load(sys.stdin); print(r.get('clone_url',''))")

    if [ -z "$clone_url" ]; then
      echo "Failed to create repository. Response:"
      echo "$create_resp"
      exit 1
    fi

    git init
    git add .
    git commit -m "Initial import"
    git branch -M main || true

    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

    export GIT_ASKPASS="$(mktemp)"
    cat > "$GIT_ASKPASS" <<EOP
#!/bin/sh
echo "$GITHUB_TOKEN"
EOP
    chmod +x "$GIT_ASKPASS"

    git push --set-upstream origin main

    rm -f "$GIT_ASKPASS"
    unset GIT_ASKPASS

    echo "Done. Repo pushed to https://github.com/${GITHUB_USER}/${REPO_NAME}"
