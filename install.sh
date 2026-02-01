#!/bin/bash
# DomainBench Installation Script for Unix/Linux/macOS
# This script installs domainbench globally using pipx (recommended)
# or falls back to pip --user installation

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
cat << 'EOF'
    ,---,                        ____                                      ,---,.                                  ,---,
  .'  .' `\                    ,'  , `.             ,--,                 ,'  .'  \                               ,--.' |
,---.'     \    ,---.       ,-+-,.' _ |           ,--.'|         ,---, ,---.' .' |               ,---,           |  |  :
|   |  .`\  |  '   ,'\   ,-+-. ;   , ||           |  |,      ,-+-. /  ||   |  |: |           ,-+-. /  |          :  :  :
:   : |  '  | /   /   | ,--.'|'   |  || ,--.--.   `--'_     ,--.'|'   |:   :  :  /   ,---.  ,--.'|'   |   ,---.  :  |  |,--.
|   ' '  ;  :.   ; ,. :|   |  ,', |  |,/       \  ,' ,'|   |   |  ,"' |:   |    ;   /     \|   |  ,"' |  /     \ |  :  '   |
'   | ;  .  |'   | |: :|   | /  | |--'.--.  .-. | '  | |   |   | /  | ||   :     \ /    /  |   | /  | | /    / ' |  |   /' :
|   | :  |  ''   | .; :|   : |  | ,    \__\/: . . |  | :   |   | |  | ||   |   . |.    ' / |   | |  | |.    ' /  '  :  | | |
'   : | /  ; |   :    ||   : |  |/     ," .--.; | '  : |__ |   | |  |/ '   :  '; |'   ;   /|   | |  |/ '   ; :__ |  |  ' | :
|   | '` ,/   \   \  / |   | |`-'     /  /  ,.  | |  | '.'||   | |--'  |   |  | ; '   |  / |   | |--'  '   | '.'||  :  :_:,'
;   :  .'      `----'  |   ;/        ;  :   .'   \;  :    ;|   |/      |   :   /  |   :    |   |/      |   :    :|  | ,'
|   ,.'                '---'         |  ,     .-./|  ,   / '---'       |   | ,'    \   \  /'---'        \   \  / `--''
'---'                                 `--`---'     ---`-'              `----'       `----'               `----'
EOF
echo -e "${NC}"

echo -e "${GREEN}DomainBench Installation Script${NC}"
echo "=================================="
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo -e "Detected OS: ${CYAN}$OS${NC}"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python is not installed. Please install Python 3.8+ first.${NC}"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "Python version: ${CYAN}$PYTHON_VERSION${NC}"

# Function to install with pipx (recommended)
install_with_pipx() {
    echo ""
    echo -e "${GREEN}Installing with pipx (recommended)...${NC}"

    # Check if pipx is installed
    if ! command -v pipx &> /dev/null; then
        echo -e "${YELLOW}pipx not found. Installing pipx first...${NC}"
        $PYTHON_CMD -m pip install --user pipx
        $PYTHON_CMD -m pipx ensurepath

        # Source the updated PATH
        if [[ "$OS" == "macos" ]]; then
            source ~/.zshrc 2>/dev/null || source ~/.bash_profile 2>/dev/null || true
        else
            source ~/.bashrc 2>/dev/null || source ~/.profile 2>/dev/null || true
        fi
    fi

    # Get the script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Install domainbench with pipx
    echo -e "Installing domainbench from ${CYAN}$SCRIPT_DIR${NC}..."
    pipx install "$SCRIPT_DIR" --force

    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo -e "Run ${CYAN}domainbench --help${NC} to get started."
}

# Function to install with pip --user
install_with_pip_user() {
    echo ""
    echo -e "${GREEN}Installing with pip --user...${NC}"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    $PYTHON_CMD -m pip install --user "$SCRIPT_DIR"

    # Determine user bin path
    USER_BIN=$($PYTHON_CMD -m site --user-base)/bin

    # Check if user bin is in PATH
    if [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
        echo ""
        echo -e "${YELLOW}Adding $USER_BIN to PATH...${NC}"

        SHELL_RC=""
        if [[ "$OS" == "macos" ]]; then
            if [[ -f ~/.zshrc ]]; then
                SHELL_RC=~/.zshrc
            else
                SHELL_RC=~/.bash_profile
            fi
        else
            SHELL_RC=~/.bashrc
        fi

        echo "" >> "$SHELL_RC"
        echo "# DomainBench PATH" >> "$SHELL_RC"
        echo "export PATH=\"\$PATH:$USER_BIN\"" >> "$SHELL_RC"

        echo -e "${YELLOW}Added to $SHELL_RC. Please run: source $SHELL_RC${NC}"
    fi

    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo -e "Run ${CYAN}domainbench --help${NC} to get started."
}

# Main installation logic
echo ""
echo "Select installation method:"
echo "  1) pipx (recommended - isolated environment, auto PATH)"
echo "  2) pip --user (installs to user site-packages)"
echo ""
read -p "Enter choice [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        install_with_pipx
        ;;
    2)
        install_with_pip_user
        ;;
    *)
        echo -e "${RED}Invalid choice. Defaulting to pipx.${NC}"
        install_with_pipx
        ;;
esac

echo ""
echo -e "${GREEN}Don't forget to set your API keys:${NC}"
echo "  export OPENAI_API_KEY=your_key"
echo "  export GEMINI_API_KEY=your_key"
echo "  export ANTHROPIC_API_KEY=your_key"
echo ""
