"""
Example user pattern configuration (~/.config/fplit/patterns.py)
Override or add to fplit's default patterns.
"""

def patterns():
    return {
        # Override default logging pattern to be more strict
        'logging_config': lambda n: (
            FplitParser._is_call_to(n, 'logging', 'basicConfig')  # Only match basicConfig
        ),
        
        # Add custom pattern for your own library
        'myapp_config': lambda n: (
            FplitParser._is_call_to(n, 'myapp', 'initialize') or
            FplitParser._is_call_to(n, 'myapp', ['config', 'setup'])
        )
    }