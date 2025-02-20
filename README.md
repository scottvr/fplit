# fplit

A Python source code file-splitting tool that intelligently separates function calls from within a single file into individual files (e.g., for tests or demo purposes) while preserving setup and calling context (or, optionally, not).

The name 'fplit' is a combination of 'file' and 'split', with a typographical nod to the historical 'long s' (ſ) character; 'split' would have appeared as 'ſplit' in historical typography.

## Overview

`fplit` reads in Python source files containing one or more function _*calls*_ (typically in a `__main__` block or at module level) and splits them into separate, self-contained files. Necessary context for successful execution of the functions (such as imports, setup code, and related statements) is preserved from the source file and added to the main execution block of each generated script. 

Or, if the source contains function _*definitions*_, and the generated files are not intended for execution, this context can be skipped via the `--funcdefs-only` option.

### Key Features

- Splits Python files into function-call-specific demonstration files
  - alternatively, into function-specific files containing only the actual function definition (e.g., for reference/documentation purposes)
- Intelligently preserves setup code and configuration (or optionally, doesn't)
- Maintains imports and necessary context (or optionally, doesn't)
- Handles both explicit `__main__` blocks and module-level code
- Smart detection and inclusion of related print statements and comments (or optionally, not)
- Configurable pattern matching for setup code detection

## Installation

```bash
git clone https://github.com/scottvr/fplit.git
cd fplit
python -m pip install -r requirements.txt
```

## Command Line Options

```
usage: fplit.py [-h] [-o OUTPUT_DIR] [-v] [--wrap-main] [--no-setup]        
                [--list-patterns]
                [--disable-patterns PATTERN [PATTERN ...]]
                [--enable-patterns PATTERN [PATTERN ...]] [--show-setup]    
                [--patterns-dir PATTERNS_DIR] [--skip-user-patterns]        
                [--skip-project-patterns]
                source_file

positional arguments:
  source_file             Python source file to split

optional arguments:
  -h, --help              show help message and exit
  -o OUTPUT_DIR           output directory for split files (default: current directory)
  -v, --verbose           increase output verbosity (use -v or -vv)
  --wrap-main             always wrap code in __main__ blocks
  --no-setup              skip preservation of module-level setup code
  --funcdefs-only         extracts only function definitions to unique files
  --show-setup            show detected module-level setup code without splitting
  --list-patterns         list all available setup patterns
  --disable-patterns      disable specific setup patterns
  --enable-patterns       enable only specified patterns
  --patterns-dir          directory containing custom pattern definitions
  --skip-user-patterns    skip loading user pattern overrides
  --skip-project-patterns skip loading project-specific patterns
  --similarity-threshold  threshold for print statement similarity
```

## Usage

Basic usage:
```bash
python fplit.py demo.py                 # Split into current directory
python fplit.py demo.py -o output_dir   # Split into specified directory
python fplit.py demo.py -v              # Show progress
python fplit.py demo.py -vv             # Show detailed debug info
```

### Example - Extracting functions into single-purpose fully-runnable scripts demonstrating one specific function

Given the following as input file `demo.py`:
```python
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set plot style
plt.style.use('seaborn')

if __name__ == "__main__":
    # Demo the data processing
    data = process_data(sample_input)
    print("Data processed successfully")
    
    # Visualize results
    plot_results(data)
    print("Generated visualization")
```

Running:
```bash
python fplit.py demo.py
```

will create separate files for each function call, preserving necessary setup, comments, and print statements:
```python
### generated file: process_data_demo.py
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    data = process_data(sample_input)
    print("Data processed successfully")
    exit(0)
```
``` python
### generated file: plot_results_demo.py
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set plot style
plt.style.use('seaborn')

if __name__ == "__main__":
    plot_results(data)
    print("Generated visualization")
    exit(0)
```

### Example - Function Reference Extraction

This mode extracts only the pure function definitions from your source code. This is useful for creating reference libraries or cataloging implementations:

Given the following as an input file:
```python
import numpy as np
from typing import List

def quicksort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def binary_search(arr: List[int], target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

if __name__ == "__main__":
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    sorted_arr = quicksort(test_arr)
    idx = binary_search(sorted_arr, 6)
```

Running:
```bash
python fplot.py source.py --funcdefs-only
```

will create separate files for each function call, _*NOT*_ preserving any surrounding setup, comments, etc:

```python
### generated file: quicksort.py
def quicksort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```
```python
### generated file: binary_search.py
def binary_search(arr: List[int], target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

This mode:
- Extracts only function definitions
- Names files directly after the functions
- Excludes imports, setup code, and main blocks
- Preserves function signatures and type hints
- Creates a clean reference library of implementations

This is particularly useful when:
- Creating an algorithm reference library
- Extracting reusable functions from existing code
- Building a catalog of implementation patterns
- Preparing code examples for documentation
 
## Setup Pattern Detection

fplit intelligently detects and preserves setup code for many popular Python libraries. Here's what each pattern matches:

### Data Science & ML
- **NumPy**: Random seeds, print options, error settings (but not array operations)
- **Pandas**: Display options, default settings (but not data operations)
- **Matplotlib**: Style settings, backend config (but not actual plotting)
- **Seaborn**: Theme setting, style config (but not visualizations)
- **Plotly**: Template selection, renderer config (but not plotting)
- **TensorFlow**: GPU/device config, random seeds (but not model ops)
- **PyTorch**: Device selection, seeds, cudnn config (but not training)
- **JAX**: Platform selection, precision config (but not computations)
- **Scikit-learn**: Random state setup (but not model operations)

### Web & API
- **FastAPI**: App creation, middleware setup (but not routes)
- **Django**: Settings adjustment (but not views)
- **Requests**: Session creation, auth setup (but not API calls)
- **SQLAlchemy**: Engine creation, pool config (but not queries)

### Testing & Debug
- **Pytest**: Skip conditions, import checking (but not test functions)
- **Logging**: Logger creation, level setting (but not log messages)
- **Warnings**: Warning filters (but not warning raises)

### Other
- **OpenCV**: Threading config, window params (but not image ops)
- **Ray**: Init config, resource setup (but not computations)
- **Random**: Seed setting (but not generation)
- **Environment Variables**: Environment variable setting

### Setup Patterns Configuration Guide
[Setup Patterns Configuration Guide](https://github.com/scottvr/fplit/blob/main/Pattern_Configuration_Guide.md)

## TODO

Contributions are welcome. Here are some things on the todo list:

- Additional setup patterns for other popular libraries
- Smarter handling of function dependencies
- Support for async/await syntax
- Configuration file support

## License

MIT
