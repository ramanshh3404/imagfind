# imagfind

`imagfind` is a complete, folder-wise, self-maintaining AI-powered semantic image search CLI tool. It uses Google's state-of-the-art **SigLIP** (via Hugging Face `transformers`) for joint image-text embeddings and **ChromaDB** for local vector storage.

---

## Key Features

- **Folder-Wise Local Indexing**: Operates relative to the current working directory (`pwd`), storing a local vector database inside `<pwd>/.imagfind_db/`. Moving or renaming parent folders does not break the index since paths are saved relatively.
- **Automatic Self-Maintenance**: Every time you run a command (search, list, diagnostics), the tool automatically scans the directory for updates (adds new files, re-embeds modified files, cleans up deleted files). If nothing changed, it runs instantly.
- **Joint Image-Text Space**: Query the database using standard text strings OR an image path as a visual template.
- **Developer-Friendly Pipeline Outputs**: Passing `--json` prints raw JSON to `stdout` while suppressing visual formatting. Spinner logs, progress bars, and Hugging Face loaders are redirected to `stderr`, allowing you to pipe query output directly into toolchains (e.g. `jq` or custom scripts).
- **Rich Terminal UI**: Displays search results, stats panels, and image indexes in high-fidelity colored terminal tables.

---

## Installation

1. Make sure you have **Python >= 3.11** installed.
2. If you use `uv`, sync and build the package locally:
   ```bash
   # Sync virtualenv dependencies
   uv sync

   # Install the package locally in editable mode
   uv pip install -e .
   ```
3. Alternatively, install via standard `pip`:
   ```bash
   pip install -e .
   ```

*Note: On your very first search or indexing run, the tool will automatically download the lightweight `google/siglip-base-patch16-224` vision-language model (~200MB) from the Hugging Face Hub and cache it locally.*

---

## Usage Examples

Run commands from the directory containing the images you want to search.

### 1. Diagnostic Info
Check database location, collection size, and execution hardware (CPU vs CUDA/GPU):
```bash
imagfind --info
```

### 2. Search Index (Text Query)
Query the local images with a descriptive text string:
```bash
imagfind -t "a cute dog playing in the grass"
```

### 3. Search Index (Image Query)
Find files similar to a template image:
```bash
imagfind -i path/to/query_image.jpg
```

### 4. Search Options (Limits, Thresholds, and JSON)
Limit output, set a similarity threshold, or extract raw JSON output:
```bash
# Return top 3 results matching threshold
imagfind -t "mountains" --limit 3 --threshold 0.05

# Pipe raw JSON results to another tool
imagfind -t "cat sleeping" --json | jq .
```

### 5. Listing All Indexed Images
Show a table of all images registered in the current directory's database:
```bash
imagfind --list
```

### 6. Recursive Scanning
Include subdirectories in your scan and search space:
```bash
imagfind -t "lake reflections" -r
```

### 7. Resetting the Index
Delete the database index for the current directory:
```bash
imagfind --clear
```

### 8. View Directory Statistics
Display detailed statistics about the current indexed folder and database size on disk:
```bash
imagfind --stats
```

### 9. Identify Duplicates or Near-Duplicates
Find duplicate or highly similar images in the index based on cosine similarity:
```bash
# Find duplicates using default 90% threshold
imagfind --duplicates

# Find duplicates using a custom threshold of 95%
imagfind --duplicates --threshold 0.95
```

### 10. Automatically Open Top Search Result
Open the single top search result using the system's default image viewer:
```bash
imagfind -t "sunset" --open
```

### 11. Generate Preview Grid
Compile the top search results into a visual grid, complete with rank overlays and similarity scores, and open it:
```bash
imagfind -t "cars" --preview --limit 6
```

### 12. Export Results to CSV
Export the ranked search results directly to a CSV file:
```bash
imagfind -t "food" --csv search_results.csv
```

