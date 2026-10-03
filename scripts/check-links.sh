#!/usr/bin/env bash
# Local link check script
# Usage: ./scripts/check-links.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running link check with lychee...${NC}"

# Check if lychee is installed
if ! command -v lychee &> /dev/null; then
    echo -e "${YELLOW}lychee not found. Installing...${NC}"
    cargo install lychee
fi

# Run lychee with config
echo -e "${GREEN}Running link check...${NC}"
lychee --config .lychee.toml .

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All links are valid!${NC}"
    exit 0
else
    echo -e "${RED}✗ Broken links found!${NC}"
    exit 1
fi