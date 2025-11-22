import json
from pathlib import Path
from anthropic import Anthropic


class SQLRetriever:
    def __init__(self):
        self.papers = None
        self.index_path = Path(__file__).parent / "papers.json"
        self.client = Anthropic()

    def index(self, papers: list[dict]):
        self.papers = papers
        with open(self.index_path, "w") as f:
            json.dump(papers, f, indent=2)

    def load(self):
        with open(self.index_path) as f:
            self.papers = json.load(f)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        prompt = f"""Given this query: "{query}"

Find the most relevant paper IDs from this list:
{json.dumps(self.papers, indent=2)}

Return only a JSON array of {k} paper IDs in order of relevance, like: ["paper_001", "paper_002"]"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        result_text = response.content[0].text.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        return json.loads(result_text)
