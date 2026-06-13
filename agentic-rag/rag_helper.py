import json
import shutil
import subprocess

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='sonnet'
    ):
        self.index = index
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append('FILE: ' + doc['filename'])
            lines.append(doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        claude_bin = shutil.which('claude') or 'claude'

        result = subprocess.run(
            [
                claude_bin,
                '-p',
                '--model', self.model,
                '--system-prompt', self.instructions,
                '--output-format', 'json',
            ],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
        )

        if result.returncode != 0:
            raise RuntimeError(
                f'claude -p failed (exit {result.returncode}): {result.stderr.strip()}'
            )

        data = json.loads(result.stdout)
        # Stash the full response so callers can inspect usage/cost afterwards.
        self.last_response = data
        self.last_usage = data.get('usage')
        return data.get('result', '').strip()

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer
