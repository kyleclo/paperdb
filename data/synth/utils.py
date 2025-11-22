import string
import unicodedata


def clean_query(query: str) -> str:
    """
    Clean a query string by normalizing it to a standard format.

    Steps:
    1. Lowercase all text
    2. Remove punctuation
    3. Normalize whitespace (collapse multiple spaces, strip leading/trailing)
    4. Convert to ASCII (remove special unicode characters)

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
    """
    # Step 1: Lowercase
    query = query.lower()

    # Step 2: Normalize unicode characters to ASCII
    # NFD = decompose unicode characters (e.g., é -> e + ´)
    # Then encode to ASCII, ignoring characters that can't be represented
    query = unicodedata.normalize('NFD', query)
    query = query.encode('ascii', 'ignore').decode('ascii')

    # Step 3: Replace punctuation with spaces
    # Create translation table that maps each punctuation character to a space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    query = query.translate(translator)

    # Step 4: Normalize whitespace
    # Split on any whitespace and rejoin with single spaces
    # This also strips leading/trailing whitespace and collapses multiple spaces
    query = ' '.join(query.split())

    return query


if __name__ == '__main__':
    # Test the function
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

    print("Testing clean_query function:\n")
    for test in test_cases:
        result = clean_query(test)
        print(f"Input:  {repr(test)}")
        print(f"Output: {repr(result)}")
        print()
