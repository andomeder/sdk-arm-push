# Load .env without overriding variables already set in the shell.
# Lines are KEY=VALUE; text after an unquoted " #" is a comment.
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%[[:space:]]#*}"
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"; val="${BASH_REMATCH[2]}"
    val="${val%"${val##*[![:space:]]}"}"
    [ -z "${!key+x}" ] && export "$key=$val"
  done < .env
fi
