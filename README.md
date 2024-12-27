# fplit

A Python script splitting tool that intelligently separates function calls from a single file into individual files (e.g., for tests or demo purposes) while preserving setup context.

Though the docs still repeatedly refer to "demonstration", it's well beyond its initial purpose of splitting an example usage file into multiple files, hence the reason I am sharing it.

The name 'fplit' is a combination of 'file' and 'split', with a typographical nod to the historical 'long s' (ſ) character; 'split' would have appeared as 'ſplit' in historical typography.

## Overview

`fplit` takes Python scripts containing multiple function demonstrations (typically in a `__main__` block or at module level) and splits them into separate, self-contained files. It intelligently preserves necessary context like imports, setup code, and related statements.

### Key Features

- Splits Python files into function-specific demonstration files
- Intelligently preserves setup code and configuration
- Maintains imports and necessary context
- Handles both explicit `__main__` blocks and module-level code
- Smart detection of related print statements and comments
- Configurable pattern matching for setup code detection

## Installation

```bash
git clone https://github.com/scottvr/fplit.git
cd fplit
python -m pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python fplit.py demo.py                 # Split into current directory
python fplit.py demo.py -o output_dir   # Split into specified directory
python fplit.py demo.py -v              # Show progress
python fplit.py demo.py -vv             # Show detailed debug info
```

### Command Line Options

```
usage: fplit [-h] [-o OUTPUT_DIR] [-v] [--wrap-main] [--no-setup] [--show-setup]
             [--list-patterns] [--disable-patterns PATTERN [PATTERN ...]]
             [--enable-patterns PATTERN [PATTERN ...]]
             source_file

positional arguments:
  source_file           Python source file to split

optional arguments:
  -h, --help           show help message and exit
  -o OUTPUT_DIR        output directory for split files (default: current directory)
  -v, --verbose        increase output verbosity (use -v or -vv)
  --wrap-main          always wrap code in __main__ blocks
  --no-setup           skip preservation of module-level setup code
  --show-setup         show detected module-level setup code without splitting
  --list-patterns      list all available setup patterns
  --disable-patterns   disable specific setup patterns
  --enable-patterns    enable only specified patterns
```

### Example

Given an input file `demo.py`:
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

Creates separate files for each function call, preserving necessary setup:
```python
# process_data_demo.py
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    data = process_data(sample_input)
    print("Data processed successfully")
    exit(0)

# plot_results_demo.py
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

## Setup Patterns Configuration Guide

## Development

Contributions are welcome! Here are some areas that could use enhancement:

- Additional setup patterns for other popular libraries
- Smart handling of function dependencies
- Support for async/await syntax
- Configuration file support
- Integration with IDE tools

## License

MIT

## Contributing

Sure!