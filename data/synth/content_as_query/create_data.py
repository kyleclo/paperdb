#!/usr/bin/env python3
"""
Generate synthetic queries from paper content using keyphrase extraction.
Extracts important phrases from abstracts and paragraphs that users would search for.
Supports multiple LLM providers: Claude, GPT, Gemini
"""

import json
import random
import sys
import os
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import clean_query


def get_keywords_prompt(content, max_phrases=5):
    """Generate prompt for keyword extraction."""
    return f"""Extract {max_phrases} keyphrases from this research paper that a user would likely use in a search query to find this paper.

Focus on:
- Named entities (methods, datasets, systems, algorithms)
- Important concepts and terminology
- Key research topics and domains
- Specific technical terms

{content}

Return ONLY a comma-separated list of keyphrases, nothing else. Example format:
neural networks, sentiment analysis, BERT, transformer architecture, text classification"""


def get_key_passages_prompt(content, max_passages=3):
    """Generate prompt for key passage extraction."""
    return f"""Extract {max_passages} key passages (1-2 sentences each) from this research paper that capture distinctive, salient aspects that help distinguish this paper from others.

Focus on passages that contain:
- Novel findings or unique results
- Specific methodological contributions
- Interesting observations or insights
- Concrete examples or applications
- Distinctive features that make this paper memorable

Avoid generic statements about the field or topic.

{content}

CRITICAL: Return ONLY the passages separated by " | ". Do NOT include any preamble, numbering, or explanations. Just the passages themselves.

Example format:
We achieve 95% accuracy on ImageNet using only 10% of the training data. | Our method reduces inference time by 3x compared to previous approaches. | The model works on both text and image data."""


def extract_keyphrases_claude(paper, client, style='keywords', max_items=5):
    """Extract keyphrases or passages using Claude."""
    from anthropic import Anthropic

    content = build_content(paper)

    if style == 'keywords':
        prompt = get_keywords_prompt(content, max_items)
        separator = ','
    else:  # key_passages
        prompt = get_key_passages_prompt(content, max_items)
        separator = '|'

    try:
        max_tokens = 500 if style == 'key_passages' else 200
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=max_tokens,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text.strip()
        items = [item.strip() for item in response_text.split(separator)]
        return items[:max_items]
    except Exception as e:
        print(f"  ⚠ Claude error: {e}")
        return fallback_keyphrases(paper)


def extract_keyphrases_gpt(paper, client, style='keywords', max_items=5):
    """Extract keyphrases or passages using GPT."""
    content = build_content(paper)

    if style == 'keywords':
        prompt = get_keywords_prompt(content, max_items)
        separator = ','
    else:  # key_passages
        prompt = get_key_passages_prompt(content, max_items)
        separator = '|'

    try:
        max_tokens = 500 if style == 'key_passages' else 200
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest GPT model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
        items = [item.strip() for item in response_text.split(separator)]
        return items[:max_items]
    except Exception as e:
        print(f"  ⚠ GPT error: {e}")
        return fallback_keyphrases(paper)


def extract_keyphrases_gemini(paper, client, style='keywords', max_items=5):
    """Extract keyphrases or passages using Gemini."""
    content = build_content(paper)

    if style == 'keywords':
        prompt = get_keywords_prompt(content, max_items)
        separator = ','
    else:  # key_passages
        prompt = get_key_passages_prompt(content, max_items)
        separator = '|'

    try:
        response = client.generate_content(prompt)
        response_text = response.text.strip()
        items = [item.strip() for item in response_text.split(separator)]
        return items[:max_items]
    except Exception as e:
        print(f"  ⚠ Gemini error: {e}")
        return fallback_keyphrases(paper)


def build_content(paper):
    """Build content string from paper for analysis."""
    content_parts = []

    if paper.get('title'):
        content_parts.append(f"Title: {paper['title']}")

    if paper.get('abstract'):
        content_parts.append(f"\nAbstract: {paper['abstract']}")

    if paper.get('paragraphs'):
        para_texts = []
        for i, para in enumerate(paper['paragraphs'][:3]):  # First 3 paragraphs
            if para.get('text'):
                para_texts.append(para['text'])
        if para_texts:
            content_parts.append(f"\nContent: {' '.join(para_texts)}")

    content = ''.join(content_parts)

    # Truncate if too long
    if len(content) > 3000:
        content = content[:3000] + "..."

    return content


def fallback_keyphrases(paper):
    """Fallback to title if LLM fails."""
    if paper.get('title'):
        return [paper['title']]
    return []


def create_content_query(paper, client, llm_type, style='keywords', num_items=None):
    """
    Create a query from extracted keyphrases or passages.

    Args:
        paper: Paper dictionary
        client: LLM client
        llm_type: Type of LLM ('claude', 'gpt', 'gemini')
        style: Extraction style ('keywords' or 'key_passages')
        num_items: Number of items to use (None = random)

    Returns:
        Query string
    """
    # Determine number of items to extract
    max_items = 5 if style == 'keywords' else 3

    # Extract items based on LLM type
    if llm_type == 'claude':
        items = extract_keyphrases_claude(paper, client, style, max_items)
    elif llm_type == 'gpt':
        items = extract_keyphrases_gpt(paper, client, style, max_items)
    elif llm_type == 'gemini':
        items = extract_keyphrases_gemini(paper, client, style, max_items)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    if not items:
        return ""

    # Randomly select a subset
    if num_items is None:
        if style == 'keywords':
            num_items = random.randint(2, min(4, len(items)))
        else:  # key_passages
            num_items = random.randint(1, min(2, len(items)))

    num_items = min(num_items, len(items))
    selected_items = random.sample(items, num_items)

    # Clean each item individually
    cleaned_items = [clean_query(item) for item in selected_items]

    # Shuffle and join with commas
    random.shuffle(cleaned_items)
    return ', '.join(cleaned_items)


def create_dataset(input_path, output_path, llm_type='claude', style='keywords', seed=42):
    """
    Create content-based synthetic query dataset.

    Args:
        input_path: Path to input papers file
        output_path: Path to output JSONL file
        llm_type: Type of LLM to use ('claude', 'gpt', 'gemini')
        style: Extraction style ('keywords' or 'key_passages')
        seed: Random seed for reproducibility
    """
    # Initialize appropriate client
    if llm_type == 'claude':
        from anthropic import Anthropic
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        client = Anthropic(api_key=api_key)
        model_name = "Claude 3 Haiku"

    elif llm_type == 'gpt':
        from openai import OpenAI
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        model_name = "GPT-4o-mini"

    elif llm_type == 'gemini':
        import google.generativeai as genai
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not set")
            sys.exit(1)
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-1.5-flash')
        model_name = "Gemini 1.5 Flash"

    else:
        print(f"ERROR: Unknown LLM type: {llm_type}")
        sys.exit(1)

    # Set random seed
    random.seed(seed)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Using {model_name} for {style} extraction\n")

    papers_processed = 0

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            paper = json.loads(line)

            # Create query from content (already cleaned within the function)
            query = create_content_query(paper, client, llm_type, style)

            if not query:
                print(f"  ⚠ Skipping paper {paper.get('paperId')} - no query generated")
                continue

            # Create output record
            output_record = {
                'query': query,
                'paperId': paper['paperId'],
                'relevance': 1
            }

            # Write to output
            outfile.write(json.dumps(output_record) + '\n')

            papers_processed += 1

            # Print progress
            if papers_processed <= 5 or papers_processed % 20 == 0:
                print(f"Processed {papers_processed} papers")
                if papers_processed <= 3:
                    print(f"  Title: {paper.get('title', 'N/A')[:60]}...")
                    print(f"  Query: {query[:80]}...")

    print(f"\n✓ Processed {papers_processed} papers successfully!")
    print(f"✓ Output saved to: {output_path}")


def main():
    input_path = Path(__file__).parent.parent.parent / 'raw' / 'papers_100.jsonl'
    output_dir = Path(__file__).parent

    # Create datasets with both LLMs and both styles
    for llm_type in ['claude', 'gpt']:
        for style in ['keywords', 'key_passages']:
            print("=" * 70)
            print(f"CREATING DATASET: {llm_type.upper()} + {style.upper()}")
            print("=" * 70)

            output_path = output_dir / f'train_{llm_type}_{style}.jsonl'

            try:
                create_dataset(input_path, output_path, llm_type=llm_type, style=style, seed=42)
            except Exception as e:
                print(f"⚠ Skipping {llm_type} {style}: {e}")
                continue

            print("\n\n")


if __name__ == '__main__':
    main()
