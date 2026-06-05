# Ticket RAG Estimator

Local MVP for estimating software task effort from similar historical tickets.

The system uses:

- CSV as the source of truth
- Ollama embeddings for retrieval
- ChromaDB as the persisted local vector database
- deterministic effort statistics for the suggested numeric estimate
- a fixed story point planning view derived from the hour estimate
- Ollama chat for explanation only

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install and run Ollama, then pull the demo models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

Optional config:

```bash
copy .env.example .env
```

## Data

The demo dataset is in:

```text
data/tasks.csv
```

The held-out evaluation dataset is in:

```text
data/evaluation_tasks.csv
```

Required columns:

```text
task_id,title,description,actual_hours
```

## Commands

Create or refresh the persisted Chroma index:

```bash
python src/main.py index
```

Check the index:

```bash
python src/main.py status
```

Estimate a new ticket:

```bash
python src/main.py estimate "Add PDF export for invoice table"
```

The estimate output shows both planning views:

```text
Time planning view:
6-10 hours
Point estimate:
8 hours

Story point planning view:
3-5 points
Point estimate:
3 points
```

Skip the LLM explanation and show deterministic evidence only:

```bash
python src/main.py estimate "Add PDF export for invoice table" --no-llm
```

Run leakage-safe evaluation:

```bash
python src/main.py evaluate
```

Evaluation uses `data/tasks.csv` as training data and `data/evaluation_tasks.csv` as held-out test data. Test tickets are excluded from the evaluation index, so the system cannot retrieve the same ticket it is trying to estimate.

Evaluation reports hour MAE and derived story point MAE for the global average, global median, and RAG median methods. Story point error is derived by converting both predicted hours and actual hours through the same fixed mapping.

The evaluation index is rebuilt on each run in:

```text
storage/evaluation_chroma
```

Use another held-out CSV:

```bash
python src/main.py evaluate --test-csv path/to/test_tasks.csv
```

If the default held-out CSV is missing, evaluation falls back to a time-ordered train/test split of `data/tasks.csv`.

Detailed evaluation rows are saved to:

```text
outputs/evaluation_results.csv
```

## Estimate Boundary

The CLI prints a suggested estimate and evidence. The user remains responsible for accepting or changing the final numeric estimate.

## Story Point Mapping

Story points are a derived planning view, not a separate learned estimate. The current fixed mapping is:

```text
<= 0.5h -> 0.5 points
<= 2h   -> 1 point
<= 4h   -> 2 points
<= 8h   -> 3 points
<= 16h  -> 5 points
<= 32h  -> 8 points
> 32h   -> 13 points
```
