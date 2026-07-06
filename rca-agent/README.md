# Standalone RCA-Agent Baseline

This directory contains only the RCA-Agent controller/executor baseline logic adapted to run against the colocated RCA-datasets tree.

## Layout

```text
/data/baselines/
  RCA-datasets/
  rca-agent/
    rca_agent/
    requirements.txt
    config.example.json
```

The agent does not import `/data/OpenRCA` and does not require the original OpenRCA date-level dataset layout.

## OpenRouter Configuration

Set the API key as an environment variable:

```bash
export OPENROUTER_API_KEY='sk-or-v1-...'
```

Model settings are read from `config.json` by default:

```json
{
  "openrouter": {
    "model": "openai/gpt-5-mini",
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": 0.0,
    "reasoning_effort": "medium",
    "seed": 0
  }
}
```

Optional environment variables:

```bash
export OPENROUTER_HTTP_REFERER='https://localhost'
export OPENROUTER_APP_TITLE='standalone-rca-agent'
```

You can also provide a JSON file:

```bash
export API_CONFIG_PATH=/data/baselines/rca-agent/config.json
```

For model, base URL, and temperature, JSON config values take precedence over environment variables. The API key is read from environment variables first.

## Usage

From `/data/baselines/rca-agent`:

```bash
python3 -m rca_agent.run --list-datasets
python3 -m rca_agent.run --dataset Bank --start-idx 0 --end-idx 0 --dry-run
python3 -m rca_agent.run --dataset Bank --start-idx 0 --end-idx 0 --sample-num 1
```

Outputs are written under:

```text
/data/baselines/rca-agent/outputs/
  result/
  monitor/
```
