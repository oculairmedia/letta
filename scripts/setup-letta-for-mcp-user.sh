#!/bin/bash
# Setup script to configure Letta Code for mcp-user

echo "Setting up Letta Code for mcp-user..."

# Create .bashrc.d directory
su - mcp-user -c "mkdir -p ~/.bashrc.d"

# Create letta setup file
su - mcp-user -c "cat > ~/.bashrc.d/letta-setup.sh << 'INNER_EOF'
# Letta Code setup
export PATH=\"/home/mcp-user/.bun/bin:\$PATH\"

# Letta environment variables
export LETTA_API_KEY=\"lettaSecurePass123\"
export LETTA_BASE_URL=\"http://192.168.50.90:8283\"

# Alias for easy access
alias letta-code='cd /opt/stacks/letta && letta'
INNER_EOF"

# Add to .bashrc if not already there
su - mcp-user -c "grep -q '.bashrc.d/letta-setup.sh' ~/.bashrc || echo 'source ~/.bashrc.d/letta-setup.sh' >> ~/.bashrc"

echo "Setup complete! To use Letta Code as mcp-user:"
echo "  1. Switch user: su - mcp-user"
echo "  2. Run: letta-code"
echo "  or: letta -p 'your prompt here'"

