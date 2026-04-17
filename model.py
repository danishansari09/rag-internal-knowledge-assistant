from openai import OpenAI

class llm:
    def __init__(self, hf_model: str, hf_key: str):
        self.hf_model = hf_model
        self.hf_key = hf_key

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.hf_key
        )

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.hf_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()