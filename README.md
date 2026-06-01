*This project has been created as part of the 42 curriculum by febraga-.*

## 📖 Description

**Call Me Maybe** is an advanced *Function Calling* engine developed for Small Language Models (SLMs). By reading prompts provided in a JSON file, the system invokes an LLM to analyze the text and extract the exact parameters required to execute a specific function.

Small-scale models (e.g. 0.6B) tend to "hallucinate" or produce invalid JSON. To address this, the inference engine uses a **Finite State Machine (FSM)** combined with a Constrained Decoding technique that intercepts the model token-by-token and restricts generation to a whitelist of allowed tokens. That guarantees 100% valid JSON output.

Generated data is passed through a final sanitizer (the "Mop") and validated with **Pydantic**, ensuring types and values are safe for production.

## Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager

### Installation
To set up the virtual environment and synchronize dependencies:

```bash
make install
```

### Execution
Run the engine with the default paths (`data/input/` and `data/output/`):

```bash
make run
```

Advanced usage (custom paths and visualizer):

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --visualize
```

## Compilation & Tests
To run linters and the test suite:

```bash
make lint
make test
```

### Clean up environment artifacts

```bash
make clean
```

## Algorithm & Architecture

The core of the engine is a token-by-token interception loop driven by `ConstrainedEngine`.

- Finite State Machine (FSM)
  - The engine tracks generation states (e.g., `NEXT_PARAM`, `VALUE`, `COMPLETED`) and exposes a strict whitelist of tokens allowed per state.

- Logit Masking
  - When an invalid token is proposed, we apply a mask `logits[mask] = -np.inf` so the model cannot select invalid tokens.

- Dynamic Pydantic Validation
  - As JSON is generated it is validated and coerced by Pydantic to guarantee correct types before persisting.

- The Output Sanitizer (the "Mop")
  - Removes semantic noise (e.g., extra words appended to SQL or over-escaped Windows paths) as a final cleanup step.

## Challenges & Resolutions

- Flake8 88-column limit: complex expressions caused linter failures.
  - Resolution: refactored heavy conditionals into clearer blocks to satisfy linting.

- Windows file path escaping: models over/under-escaped backslashes.
  - Resolution: implemented strict path-cleaning in the "Mop" with tests.

- ModuleNotFoundError during testing: pytest couldn't locate `src` when run in some environments.
  - Resolution: run tests with `python -m pytest` to ensure project root is in `sys.path`.

- Terminal formatting for debugging: hard to distinguish FSM tokens from model values.
  - Resolution: ANSI color visualization separates structural tokens from generated values.

## Performance

- **Accuracy:** handles complex prompts, SQL extraction and path parsing.
- **Reliability:** FSM guarantees valid JSON output.
- **Speed:** Processes standard test batches efficiently, keeping execution times well under the 5-minute threshold on standard hardware.
- **Code quality:** fully typed (mypy) and PEP8-compliant (flake8).

## Design Decisions

- The "Mop" (sanitizer) pattern decouples structural generation from semantic cleanup.
- Strict typing with Pydantic ensures runtime validation after the FSM guarantees well-formed JSON.
- Visual feedback via `--visualize` (ANSI colors) improves debugging.

## Testing Strategy

Unit tests focus on the Output Formatter/Sanitizer and common hallucination patterns:

- Double-escaped Windows paths (e.g. `C:\\Users\\...`).
- Truncating semantic noise appended to SQL queries.
- Handling edge cases such as empty parameter dictionaries.

These tests verify the recovery logic prior to running full end-to-end inference.

## Example Usage

Input prompt:

```
Read C:\\Users\\john\\config.ini with latin-1 encoding
```

Generated output (JSON):

```json
{
  "prompt": "Read C:\\Users\\john\\config.ini with latin-1 encoding",
  "name": "read_file",
  "parameters": {
    "path": "C:\\Users\\john\\config.ini",
    "encoding": "latin-1"
  }
}
```

## Resources

- Python 3.10 Official Documentation
- Pydantic Documentation
- `uv` package manager

## AI Usage

AI was used as a consultative tool for brainstorming, debugging and documentation. Its contributions included:

- Code linting guidance and refactoring suggestions to meet PEP8 constraints.
- Help drafting and polishing documentation.
- Brainstorming edge cases and testing strategies for the sanitizer and FSM.

Note: the core FSM architecture, token masking logic and inference algorithms were designed and implemented by the project authors.
