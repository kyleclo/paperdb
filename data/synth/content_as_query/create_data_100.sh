#!/bin/bash

# Script to generate content-based synthetic queries using LLM extraction
# Creates multiple train files using Claude and GPT with two extraction styles

set -e  # Exit on error

echo "=================================================="
echo "Generating content-based synthetic queries"
echo "Using Claude (Haiku) and GPT (4o-mini)"
echo "Styles: keywords and key_passages"
echo "=================================================="
echo ""

# Check for API keys
api_keys_found=0

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠ Warning: ANTHROPIC_API_KEY not set (will skip Claude)"
else
    echo "✓ ANTHROPIC_API_KEY found"
    api_keys_found=$((api_keys_found + 1))
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠ Warning: OPENAI_API_KEY not set (will skip GPT)"
else
    echo "✓ OPENAI_API_KEY found"
    api_keys_found=$((api_keys_found + 1))
fi

if [ $api_keys_found -eq 0 ]; then
    echo ""
    echo "ERROR: No API keys found. Please set at least one:"
    echo "  export ANTHROPIC_API_KEY=your_key"
    echo "  export OPENAI_API_KEY=your_key"
    exit 1
fi

echo ""

# Navigate to project root
cd "$(dirname "$0")/../../.."

# Run the dataset creation script
python data/synth/content_as_query/create_data.py

echo ""
echo "=================================================="
echo "Done! Generated files:"
echo "  - data/synth/content_as_query/train_claude_keywords.jsonl"
echo "  - data/synth/content_as_query/train_claude_key_passages.jsonl"
echo "  - data/synth/content_as_query/train_gpt_keywords.jsonl"
echo "  - data/synth/content_as_query/train_gpt_key_passages.jsonl"
echo "=================================================="
