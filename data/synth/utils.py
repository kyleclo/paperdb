import string
import unicodedata


def clean_query(query: str) -> str:
    """
    Clean a query string by normalizing it to a standard format.

    Steps:
    1. Lowercase all text
    2. Convert to ASCII (remove special unicode characters)
    3. Remove apostrophes (without adding whitespace)
    4. Replace remaining punctuation with spaces
    5. Normalize whitespace (collapse multiple spaces, strip leading/trailing)

    Args:
        query: The input query string

    Returns:
        A cleaned query string

    Examples:
        >>> clean_query("Hello, World!")
        'hello world'
        >>> clean_query("  Multiple   spaces  ")
        'multiple spaces'
        >>> clean_query("Café résumé")
        'cafe resume'
        >>> clean_query("two-factor")
        'two factor'
        >>> clean_query("3D-Auth: Test")
        '3d auth test'
        >>> clean_query("don't")
        'dont'
        >>> clean_query("it's a test")
        'its a test'
    """
    # Step 1: Lowercase
    query = query.lower()

    # Step 2: Normalize unicode characters to ASCII
    # NFD = decompose unicode characters (e.g., é -> e + ´)
    # Then encode to ASCII, ignoring characters that can't be represented
    query = unicodedata.normalize('NFD', query)
    query = query.encode('ascii', 'ignore').decode('ascii')

    # Step 3: Remove apostrophes without adding whitespace
    query = query.replace("'", "")

    # Step 4: Replace remaining punctuation with spaces
    # Create translation table that maps each punctuation character to a space
    # (apostrophe already removed, so exclude it)
    punctuation_without_apostrophe = string.punctuation.replace("'", "")
    translator = str.maketrans(punctuation_without_apostrophe, ' ' * len(punctuation_without_apostrophe))
    query = query.translate(translator)

    # Step 5: Normalize whitespace
    # Split on any whitespace and rejoin with single spaces
    # This also strips leading/trailing whitespace and collapses multiple spaces
    query = ' '.join(query.split())

    return query


def textual_overlap(query: str, title: str, overlap: float = 1.0, order: bool = False) -> bool:
    """
    Calculate whether there is sufficient textual overlap between query and title.

    Args:
        query: The query string
        title: The title string
        overlap: Threshold in [0.0, 1.0] for proportion of query words that must match
                 0.0 = zero overlap required (always match)
                 1.0 = all query words must be in title
        order: Whether to consider word order
               False = bag-of-words matching (more lenient)
               True = subsequence matching (stricter, preserves order)

    Returns:
        True if the overlap threshold is met, False otherwise

    Examples:
        >>> textual_overlap("machine learning", "Deep Learning", overlap=0.5, order=False)
        True  # "learning" is in title, 50% match
        >>> textual_overlap("machine learning", "Deep Learning", overlap=1.0, order=False)
        False  # only "learning" matches, not 100%
        >>> textual_overlap("deep learning model", "Deep Learning for NLP", overlap=0.66, order=True)
        True  # "deep learning" appear in order (66% match)
        >>> textual_overlap("learning deep model", "Deep Learning for NLP", overlap=0.66, order=True)
        False  # "learning deep" don't appear in that order
    """
    # Clean both strings
    query_clean = clean_query(query)
    title_clean = clean_query(title)

    # Split into words
    query_words = query_clean.split()
    title_words = title_clean.split()

    # Handle edge cases
    if not query_words:
        return True  # Empty query always matches
    if not title_words:
        return overlap == 0.0  # Empty title only matches if overlap is 0

    if order:
        # Subsequence matching: find longest common subsequence
        matched_count = longest_common_subsequence_length(query_words, title_words)
    else:
        # Bag-of-words matching: count how many query words appear in title
        title_word_set = set(title_words)
        matched_count = sum(1 for word in query_words if word in title_word_set)

    # Calculate actual overlap
    actual_overlap = matched_count / len(query_words)

    return actual_overlap >= overlap


def longest_common_subsequence_length(seq1: list, seq2: list) -> int:
    """
    Calculate the length of the longest common subsequence between two sequences.

    Args:
        seq1: First sequence (list of items)
        seq2: Second sequence (list of items)

    Returns:
        Length of the longest common subsequence

    Examples:
        >>> longest_common_subsequence_length(['a', 'b', 'c'], ['a', 'x', 'b', 'c'])
        3  # 'a', 'b', 'c' all appear in order
        >>> longest_common_subsequence_length(['a', 'c', 'b'], ['a', 'b', 'c'])
        2  # 'a', 'b' or 'a', 'c' appear in order (but not 'a', 'c', 'b')
    """
    m, n = len(seq1), len(seq2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


if __name__ == '__main__':
    # Test clean_query function
    print("=" * 60)
    print("Testing clean_query function:")
    print("=" * 60 + "\n")

    test_cases = [
        "Hello, World!",
        "  Multiple   spaces  ",
        "Café résumé",
        "Test—with—dashes",
        "UPPERCASE and lowercase",
        "Punctuation!!! @#$ Removal???",
        "Unicode: café, naïve, résumé, Zürich",
        "Mixed:   Tabs\t\tand  \n Newlines",
    ]

    for test in test_cases:
        result = clean_query(test)
        print(f"Input:  {repr(test)}")
        print(f"Output: {repr(result)}")
        print()

    # Test textual_overlap function
    print("\n" + "=" * 60)
    print("Testing textual_overlap function:")
    print("=" * 60 + "\n")

    overlap_tests = [
        # (query, title, overlap, order, expected_result, description)
        ("machine learning", "Deep Learning", 0.5, False, True, "50% match, bag-of-words"),
        ("machine learning", "Deep Learning", 1.0, False, False, "only 50% match, need 100%"),
        ("deep learning", "Deep Learning for NLP", 1.0, False, True, "100% match, bag-of-words"),
        ("deep learning model", "Deep Learning for NLP", 0.66, True, True, "deep+learning in order (66%)"),
        ("learning deep model", "Deep Learning for NLP", 0.66, True, False, "learning+deep not in order"),
        ("neural network", "Neural Network Training", 1.0, False, True, "exact match 100%"),
        ("CHI 2020", "CHI Conference 2020", 1.0, False, True, "all words present"),
        ("ACL EMNLP", "ACL 2020", 0.5, False, True, "50% match (ACL only)"),
        ("ACL EMNLP", "ACL 2020", 1.0, False, False, "only 50%, need 100%"),
        ("", "Any Title", 0.5, False, True, "empty query always matches"),
    ]

    for query, title, overlap, order, expected, desc in overlap_tests:
        result = textual_overlap(query, title, overlap=overlap, order=order)
        status = "✓" if result == expected else "✗"
        print(f"{status} Query: '{query}' | Title: '{title}'")
        print(f"  overlap={overlap}, order={order}")
        print(f"  Result: {result} | Expected: {expected} | {desc}")
        print()
