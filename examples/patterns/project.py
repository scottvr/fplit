"""
Example project pattern configuration (./patterns.py)
Define patterns specific to your project's needs.
"""

def patterns():
    return {
        # Project-specific database configuration
        'database_setup': lambda n: (
            FplitParser._is_call_to(n, 'db', 'init_app') or
            FplitParser._is_call_to(n, 'db', ['create_all'])
        ),
        
        # Custom logging setup for this project
        'logging_config': lambda n: (
            FplitParser._is_call_to(n, 'logging', 'basicConfig') or
            FplitParser._is_call_to(n, 'logging', ['getLogger', 'setLevel']) or
            # Project-specific log configuration
            FplitParser._is_call_to(n, 'app.log', 'setup')
        )
    }