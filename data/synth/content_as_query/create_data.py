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

        # Return items and token usage
        token_usage = {
            'input_tokens': message.usage.input_tokens,
            'output_tokens': message.usage.output_tokens
        }
        return items[:max_items], token_usage
    except Exception as e:
        print(f"  ⚠ Claude error: {e}")
        return fallback_keyphrases(paper), {'input_tokens': 0, 'output_tokens': 0}


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

        # Return items and token usage
        token_usage = {
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens
        }
        return items[:max_items], token_usage
    except Exception as e:
        print(f"  ⚠ GPT error: {e}")
        return fallback_keyphrases(paper), {'input_tokens': 0, 'output_tokens': 0}


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

        # Return items and token usage
        token_usage = {
            'input_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            'output_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        }
        return items[:max_items], token_usage
    except Exception as e:
        print(f"  ⚠ Gemini error: {e}")
        return fallback_keyphrases(paper), {'input_tokens': 0, 'output_tokens': 0}


def build_content(paper):
    """Build content string from paper for analysis."""
    content_parts = []

    # Use fulltext_title if available, otherwise fall back to title
    title = paper.get('fulltext_title') or paper.get('title')
    if title:
        content_parts.append(f"Title: {title}")

    # Use fulltext_abstract if available, otherwise fall back to abstract
    abstract = paper.get('fulltext_abstract') or paper.get('abstract')
    if abstract:
        content_parts.append(f"\nAbstract: {abstract}")

    # Use fulltext_body for content (first 1500 chars to leave room for title/abstract)
    if paper.get('fulltext_body'):
        body = paper['fulltext_body']
        # Take first 1500 chars of body
        body_excerpt = body[:1500]
        content_parts.append(f"\nContent: {body_excerpt}")

    content = ''.join(content_parts)

    # Truncate if too long (should be ~3000 total)
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
        Tuple of (query string, token_usage dict)
    """
    # Determine number of items to extract
    max_items = 5 if style == 'keywords' else 3

    # Extract items based on LLM type
    if llm_type == 'claude':
        items, token_usage = extract_keyphrases_claude(paper, client, style, max_items)
    elif llm_type == 'gpt':
        items, token_usage = extract_keyphrases_gpt(paper, client, style, max_items)
    elif llm_type == 'gemini':
        items, token_usage = extract_keyphrases_gemini(paper, client, style, max_items)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    if not items:
        return "", token_usage

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
    return ', '.join(cleaned_items), token_usage


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
    import time

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

    # Create tokens file path
    tokens_path = output_path.parent / f'{output_path.stem}_tokens.jsonl'

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Tokens file: {tokens_path}")
    print(f"Using {model_name} for {style} extraction\n")

    # Count total papers first
    total_papers = 0
    with open(input_path, 'r') as f:
        for _ in f:
            total_papers += 1

    papers_processed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    with open(input_path, 'r') as infile, \
         open(output_path, 'w') as outfile, \
         open(tokens_path, 'w') as tokens_file:

        for line in infile:
            paper = json.loads(line)
            corpus_id = paper.get('corpusid') or paper.get('paperId')

            # Create query from content (already cleaned within the function)
            query, token_usage = create_content_query(paper, client, llm_type, style)

            if not query:
                print(f"  ⚠ Skipping paper {corpus_id} - no query generated")
                continue

            # Create output record
            output_record = {
                'query': query,
                'corpus_id': corpus_id,
                'relevance': 1
            }

            # Write to output
            outfile.write(json.dumps(output_record) + '\n')

            # Write token usage
            token_record = {
                'corpus_id': corpus_id,
                'input_tokens': token_usage['input_tokens'],
                'output_tokens': token_usage['output_tokens']
            }
            tokens_file.write(json.dumps(token_record) + '\n')

            # Update totals
            total_input_tokens += token_usage['input_tokens']
            total_output_tokens += token_usage['output_tokens']

            papers_processed += 1

            # Print progress
            elapsed = time.time() - start_time
            papers_per_sec = papers_processed / elapsed if elapsed > 0 else 0
            eta_seconds = (total_papers - papers_processed) / papers_per_sec if papers_per_sec > 0 else 0

            if papers_processed <= 3 or papers_processed % 10 == 0 or papers_processed == total_papers:
                progress_pct = (papers_processed / total_papers) * 100
                print(f"[{papers_processed}/{total_papers}] {progress_pct:.1f}% | "
                      f"ETA: {eta_seconds:.0f}s | "
                      f"Tokens: {total_input_tokens} in, {total_output_tokens} out")
                if papers_processed <= 3:
                    print(f"  Title: {paper.get('title', 'N/A')[:60]}...")
                    print(f"  Query: {query[:80]}...")

    print(f"\n✓ Processed {papers_processed} papers successfully!")
    print(f"✓ Output saved to: {output_path}")
    print(f"✓ Tokens saved to: {tokens_path}")
    print(f"✓ Total tokens: {total_input_tokens} input, {total_output_tokens} output")


def main():
    # Paths (relative to script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    input_path = project_root / 'raw' / 'dblp-nlp-ml-ai-oa-recent-with-fulltext-tagged-100.jsonl'
    output_dir = script_dir

    # Create datasets with both LLMs and both styles
    # Set to False to skip (e.g., if insufficient credits)
    use_claude = False
    use_gpt = True

    llm_types = []
    if use_claude:
        llm_types.append('claude')
    if use_gpt:
        llm_types.append('gpt')

    for llm_type in llm_types:
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
