# One-Shot Generation

Scripts and resources for generating synthetic Bulwer-Lytton sentences using large language models.

## Files

- **`user-prompt-single.txt`** - Template for the one-shot prompt that was fed to LLMs. Contains instructions for the Bulwer-Lytton Fiction Contest rules and the original example, with a `{genre}` placeholder for substitution.

- **`prompt_genres_list.txt`** - List of 1,000 genres used to generate diverse synthetic sentences. Each genre is substituted into the `{genre}` placeholder in the one-shot prompt.

- **`api-generate.py`** - Script to query the Together API and generate responses from DeepSeek-V3.1. Used to produce 1,000 DeepSeek-generated sentences.

- **`batch_response_generate.py`** - Script to batch query OpenAI API and get responses from GPT-4.1 and GPT-5 models. Used to produce 1,000 sentences from each model.

- **`local-generate.py`** - Script to generate responses locally from `gpt-oss-120b` model on a workstation with GPUs. Used to produce 1,000 `gpt-oss-120b` generated sentences.

Generated sentences are stored in `../data/` with filenames indicating the source model (e.g., `deepseek_generated_sentences.txt`, `gpt5_generated_sentences.txt`).
