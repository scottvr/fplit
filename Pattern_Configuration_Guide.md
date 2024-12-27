# Fplit Pattern Configuration Guide

## Overview

Fplit uses pattern matching to identify setup code that should be preserved across split files rather than generating separate demo files. This guide explains how to customize these patterns.

## Pattern File Structure

Pattern files are Python modules that define a dictionary of patterns. Each pattern is a function that examines AST nodes to identify setup code.

### Basic Pattern File Example

```python
# patterns.py

def patterns():
    return {
        'my_library_config': lambda n: (
            FplitParser._is_call_to(n, 'mylibrary', 'configure') or
            FplitParser._is_call_to(n, 'mylibrary', ['setup', 'init'])
        )
    }
```

## Pattern Locations

Fplit looks for patterns in three locations, in order of precedence:

1. Built-in patterns (shipped with fplit)
2. User patterns (`~/.config/fplit/patterns.py`)
3. Project patterns (specified via `--patterns-dir`)

Later patterns override earlier ones with the same name.

## Helper Methods

Fplit provides several helper methods for pattern matching:

### _is_call_to(node, module, methods)

Matches function calls, including method chains.

```python
# Single method call
_is_call_to(node, 'logging', 'basicConfig')  # matches: logging.basicConfig()

# Method chain
_is_call_to(node, 'logging', ['getLogger', 'setLevel'])  # matches: logging.getLogger().setLevel()
```

### _is_attr_assign(node, module, attr)

Matches attribute assignments.

```python
_is_attr_assign(node, 'torch', 'backends.cudnn.deterministic')  # matches: torch.backends.cudnn.deterministic = True
```

### _is_stored_object_assignment(node, module, factory_method)

Matches when an object is created and stored.

```python
_is_stored_object_assignment(node, 'requests', 'Session')  # matches: session = requests.Session()
```

## Pattern Examples

### Simple Configuration Pattern

```python
# Match calls to my_library.setup()
'my_library_config': lambda n: FplitParser._is_call_to(n, 'my_library', 'setup')
```

### Method Chain Pattern

```python
# Match library.create().configure().initialize()
'complex_setup': lambda n: FplitParser._is_call_to(n, 'library', ['create', 'configure', 'initialize'])
```

### Multiple Setup Methods

```python
'data_config': lambda n: (
    FplitParser._is_call_to(n, 'data', 'set_format') or
    FplitParser._is_call_to(n, 'data', ['config', 'initialize']) or
    FplitParser._is_attr_assign(n, 'data', 'default_format')
)
```

### Real-world Example: Database Setup

```python
'database_config': lambda n: (
    # Match direct configuration
    FplitParser._is_call_to(n, 'db', 'configure') or
    
    # Match SQLAlchemy engine setup
    FplitParser._is_stored_object_assignment(n, 'create_engine', 'create_engine') or
    
    # Match connection pool configuration
    FplitParser._is_method_on_stored_object(n, ['pool_size', 'pool_timeout', 'pool_recycle']) or
    
    # Match environment setup
    FplitParser._is_call_to(n, 'db', ['environment', 'setup'])
)
```

## Command Line Options

Control pattern loading with these options:

```bash
# Use custom pattern directory
fplit demo.py --patterns-dir=/path/to/patterns

# Skip user patterns
fplit demo.py --skip-user-patterns

# Skip project patterns
fplit demo.py --skip-project-patterns

# Show pattern matching details
fplit demo.py -vv
```

## Debugging Patterns

Use increased verbosity to debug pattern matching:

```bash
# Show which patterns are active
fplit demo.py -v

# Show detailed pattern matching information
fplit demo.py -vv
```

Output will show:
- Active patterns
- Pattern matches attempted
- Which patterns matched or didn't match
- Any errors in pattern matching

## Best Practices

1. **Be Specific**: Pattern names should clearly indicate what they match
2. **Document Boundaries**: Comment what should and shouldn't be matched
3. **Test Patterns**: Use verbose mode to verify pattern matching
4. **Use Helper Methods**: Prefer helper methods over raw AST matching
5. **Consider Override Order**: Remember that project patterns override user patterns

## Common Pitfalls

1. **Over-matching**: Patterns that are too broad may catch unintended code
2. **Under-matching**: Missing method chains or variations in how a library is used
3. **Name Conflicts**: Pattern names should be unique or intentionally override
4. **Complex AST**: Raw AST matching can be fragile; use helper methods when possible