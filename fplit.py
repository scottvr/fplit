#!/usr/bin/env python3
"""
fplit - A Python Module Splitting Tool

The name 'fplit' is a playful combination of 'file' and 'split',
with a typographical nod to the historical 'long s' (ſ) character -
just as 'split' would have appeared as 'ſplit' in historical typography,
we present 'fplit' as a tool for splitting Python files.

This tool intelligently splits Python modules containing multiple function
demonstrations in a __main__ block into separate files, preserving context,
imports, and related code for each function.
"""

import ast
import astor
from pathlib import Path
import re
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import difflib
import traceback

@dataclass
class PrintContext:
    """Stores context about a print statement for smart grouping."""
    node: ast.Expr
    text: str
    similarity_to_function: float
    line_number: int

class FplitParser:
    @classmethod
    def _is_call_to(cls, node, module, func=None):
        """Helper to check if node is a call to module.func()"""
        return (isinstance(node, ast.Expr) and 
               isinstance(node.value, ast.Call) and
               isinstance(node.value.func, ast.Attribute) and
               isinstance(node.value.func.value, ast.Name) and
               node.value.func.value.id == module and
               (func is None or node.value.func.attr == func))
    
    @classmethod
    def _is_attr_assign(cls, node, module, attr=None):
        """Helper to check if node assigns to module.attr"""
        return (isinstance(node, ast.Assign) and
               len(node.targets) == 1 and
               isinstance(node.targets[0], ast.Attribute) and
               isinstance(node.targets[0].value, ast.Name) and
               node.targets[0].value.id == module and
               (attr is None or node.targets[0].attr == attr))
    
    @classmethod
    def _is_method_on_stored_object(cls, node, method_names: List[str]) -> bool:
        """Check if this is a method call on a stored object."""
        return (isinstance(node, ast.Expr) and
                isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and
                node.value.func.attr in method_names and
                isinstance(node.value.func.value, ast.Name))
    
    @classmethod
    def _is_stored_object_assignment(cls, node, module: str, factory_method: str) -> bool:
        """Check if this assigns the result of a module's factory method to a variable."""
        return (isinstance(node, ast.Assign) and
                isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and
                isinstance(node.value.func.value, ast.Name) and
                node.value.func.value.id == module and
                node.value.func.attr == factory_method)
    
    setup_patterns = {
        # Logging Configuration
        # MATCH: Logger creation and level setting, propagation config
        # SKIP: Actual log message calls
        'logging_config': lambda n: (
            FplitParser._is_call_to(n, 'logging', 'basicConfig') or
            FplitParser._is_call_to(n, 'logging', 'setLevel') or
            FplitParser._is_call_to(n, 'logging', 'getLogger') or
            FplitParser._is_stored_object_assignment(n, 'logging', 'getLogger') or
            FplitParser._is_method_on_stored_object(n, ['setLevel', 'propagate'])
        ),

        # Matplotlib/Pyplot Setup
        # MATCH: Style settings, backend config, default figure params
        # SKIP: Actual plotting calls, data visualization, saving figures
        'matplotlib_config': lambda n: (
            FplitParser._is_call_to(n, 'matplotlib', 'use') or
            FplitParser._is_call_to(n, 'plt', 'style.use') or
            (isinstance(n, ast.Assign) and  # rcParams assignment
             isinstance(n.targets[0], ast.Subscript) and
             isinstance(n.targets[0].value, ast.Attribute) and
             isinstance(n.targets[0].value.value, ast.Name) and
             n.targets[0].value.value.id == 'plt' and
             n.targets[0].value.attr == 'rcParams')
        ),

        # Pandas Configuration
        # MATCH: Display options, default settings
        # SKIP: Actual data operations, DataFrame creation/manipulation
        'pandas_config': lambda n: (
            FplitParser._is_call_to(n, 'pd', 'set_option') or
            FplitParser._is_call_to(n, 'pd', 'options.display')
        ),

        # Requests Setup
        # MATCH: Session creation and base config (auth, headers)
        # SKIP: Actual API calls, request sending
        'requests_config': lambda n: (
            FplitParser._is_stored_object_assignment(n, 'requests', 'Session') or
            (FplitParser._is_method_on_stored_object(n, ['auth', 'headers', 'verify', 'cert']) and
             not FplitParser._is_method_on_stored_object(n, ['get', 'post', 'put', 'delete', 'patch']))
        ),

        # NumPy
        'numpy_config': lambda n: (
            FplitParser._is_call_to(n, 'np', 'set_printoptions') or
            FplitParser._is_call_to(n, 'np', 'random.seed.')
        ),
        
        # Pandas
        'pandas_options': lambda n: FplitParser._is_call_to(n, 'pd', 'set_option'),
        
        # Matplotlib
        'matplotlib_backend': lambda n: FplitParser._is_call_to(n, 'matplotlib', 'use'),
        'plt_style': lambda n: FplitParser._is_call_to(n, 'plt', 'style.use'),
        'plt_rcparams': lambda n: isinstance(n, ast.Assign) and 
                       isinstance(n.targets[0], ast.Subscript) and
                       isinstance(n.targets[0].value, ast.Attribute) and
                       isinstance(n.targets[0].value.value, ast.Name) and
                       n.targets[0].value.value.id == 'plt' and
                       n.targets[0].value.attr == 'rcParams',
        
        # TensorFlow Setup
        # MATCH: GPU/device config, random seeds, basic TF settings
        # SKIP: Model creation, training, inference
        'tf_config': lambda n: (
            FplitParser._is_call_to(n, 'tf', 'config.set_visible_devices') or
            FplitParser._is_call_to(n, 'tf', 'random.set_seed') or
            FplitParser._is_call_to(n, 'tf', 'config.experimental.enable_op_determinism')
        ),

        # PyTorch Setup
        # MATCH: Device selection, random seeds, cudnn config
        # SKIP: Model ops, tensor operations, training
        'torch_config': lambda n: (
            FplitParser._is_call_to(n, 'torch', 'manual_seed') or
            FplitParser._is_call_to(n, 'torch', 'cuda.set_device') or
            FplitParser._is_attr_assign(n, 'torch', 'backends.cudnn.deterministic')
        ),

        # SQLAlchemy Setup
        # MATCH: Engine creation, connection pool config
        # SKIP: Actual queries, table operations, schema work
        'sqlalchemy_config': lambda n: (
            FplitParser._is_stored_object_assignment(n, 'create_engine', 'create_engine') or
            FplitParser._is_method_on_stored_object(n, ['pool_size', 'pool_timeout', 'pool_recycle'])
        ),

        # FastAPI Setup
        # MATCH: App creation, middleware setup, basic config
        # SKIP: Route definitions, actual endpoint handlers
        'fastapi_config': lambda n: (
            FplitParser._is_stored_object_assignment(n, 'FastAPI', 'FastAPI') or
            FplitParser._is_method_on_stored_object(n, ['add_middleware', 'include_router']) and
            not FplitParser._is_method_on_stored_object(n, ['get', 'post', 'put', 'delete'])
        ),

        # Seaborn Setup
        # MATCH: Theme setting, style config, default parameters
        # SKIP: Actual plot creation, data visualization
        'seaborn_config': lambda n: (
            FplitParser._is_call_to(n, 'sns', 'set_theme') or
            FplitParser._is_call_to(n, 'sns', 'set_style') or
            FplitParser._is_call_to(n, 'sns', 'set_context') or
            FplitParser._is_call_to(n, 'sns', 'set_palette')
        ),

        # Plotly Setup
        # MATCH: Template selection, renderer config
        # SKIP: Figure creation, actual plotting
        'plotly_config': lambda n: (
            FplitParser._is_attr_assign(n, 'pio', 'templates.default') or
            FplitParser._is_call_to(n, 'pio', 'renderers.default') or
            FplitParser._is_call_to(n, 'pio', 'set_config')
        ),

        # NumPy Setup
        # MATCH: Random seed, print options, error settings
        # SKIP: Actual array operations, computations
        'numpy_config': lambda n: (
            FplitParser._is_call_to(n, 'np', 'random.seed') or
            FplitParser._is_call_to(n, 'np', 'set_printoptions') or
            FplitParser._is_call_to(n, 'np', 'seterr') or
            FplitParser._is_call_to(n, 'np', 'seterrcall')
        ),

        # Pytest Setup
        # MATCH: Skip conditions, import checking, config
        # SKIP: Actual test functions, assertions
        'pytest_config': lambda n: (
            FplitParser._is_call_to(n, 'pytest', 'skip_if') or
            FplitParser._is_call_to(n, 'pytest', 'importorskip') or
            FplitParser._is_call_to(n, 'pytest', 'mark.skipif') or
            FplitParser._is_call_to(n, 'pytest', 'fixture')
        ),

        # JAX Setup
        # MATCH: Platform selection, precision config
        # SKIP: Actual computations, transformations
        'jax_config': lambda n: (
            FplitParser._is_call_to(n, 'jax', 'config.update') or
            FplitParser._is_call_to(n, 'jax', 'disable_jit') or
            FplitParser._is_attr_assign(n, 'jax', 'config.x64_enabled')
        ),

        # OpenCV Setup
        # MATCH: Threading config, window params
        # SKIP: Actual image operations, video capture
        'cv2_config': lambda n: (
            FplitParser._is_call_to(n, 'cv2', 'setNumThreads') or
            FplitParser._is_call_to(n, 'cv2', 'setUseOptimized') or
            FplitParser._is_call_to(n, 'cv2', 'namedWindow')
        ),
        

        'sklearn_random': lambda n: isinstance(n, ast.Assign) and
                        any(isinstance(n.value, ast.Call) and
                            isinstance(n.value.func, ast.Attribute) and
                            n.value.func.attr == 'check_random_state'
                            for target in n.targets),

        # Scikit-learn Setup
        # MATCH: Random state initialization, verbosity settings
        # SKIP: Actual model creation, fitting, prediction
        'sklearn_random': lambda n: (
            FplitParser._is_stored_object_assignment(n, 'sklearn.utils', 'check_random_state') or
            FplitParser._is_call_to(n, 'sklearn', 'set_config') or
            # Handle both direct calls and stored instances
            (isinstance(n, ast.Assign) and
             isinstance(n.value, ast.Call) and
             isinstance(n.value.func, ast.Attribute) and
             n.value.func.attr == 'check_random_state')
        ),

        # Keras Setup
        # MATCH: Backend config, image format, device settings
        # SKIP: Model definition, training, prediction
        'keras_config': lambda n: (
            FplitParser._is_call_to(n, 'K', 'set_image_data_format') or
            FplitParser._is_call_to(n, 'K', 'set_floatx') or
            FplitParser._is_call_to(n, 'K', 'clear_session') or
            # Handle stored backend configurations
            FplitParser._is_method_on_stored_object(n, ['set_image_data_format', 'set_floatx', 'clear_session'])
        ),
        

        # Ray Setup
        # MATCH: Init config, resource setup
        # SKIP: Actual distributed computations
        'ray_config': lambda n: (
            FplitParser._is_call_to(n, 'ray', 'init') or
            FplitParser._is_call_to(n, 'ray', 'shutdown') or
            FplitParser._is_method_on_stored_object(n, ['environment'])
        ),

        # Django Setup
        # MATCH: Settings adjustment, middleware config
        # SKIP: View definitions, URL patterns
        'django_config': lambda n: (
            FplitParser._is_attr_assign(n, 'settings') or
            isinstance(n, ast.Call) and
            isinstance(n.func, ast.Name) and
            n.func.id == 'configure'
        ),

        # Warning Configuration
        # MATCH: Warning filters, warning config
        # SKIP: Actual warning raises
        'warnings_config': lambda n: (
            FplitParser._is_call_to(n, 'warnings', 'filterwarnings') or
            FplitParser._is_call_to(n, 'warnings', 'simplefilter') or
            FplitParser._is_call_to(n, 'warnings', 'resetwarnings')
        ),

        # Random Setup
        # MATCH: Seed setting only
        # SKIP: Actual random number generation
        'random_config': lambda n: (
            FplitParser._is_call_to(n, 'random', 'seed')
        ),

        # Environment Variables
        # MATCH: Environment variable setting
        # SKIP: Environment variable reading/usage
        'environ': lambda n: (
            isinstance(n, ast.Assign) and
            isinstance(n.targets[0], ast.Subscript) and
            isinstance(n.targets[0].value, ast.Attribute) and
            isinstance(n.targets[0].value.value, ast.Name) and
            n.targets[0].value.value.id == 'os' and
            n.targets[0].value.attr == 'environ'
        )
    }

    def __init__(self, source_file: str, 
             disabled_patterns: Optional[List[str]] = None,
             enabled_patterns: Optional[List[str]] = None):
        self.source_file = Path(source_file)
        with open(source_file, 'r') as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)
        self.global_vars = {}
        self.setup_nodes = []
        self.imports = []
        self.comments = {}  # line_number -> comment text
        self._extract_comments()
        self._extract_imports()
        self._extract_global_vars()
        self.available_patterns = set(self.setup_patterns.keys())
        self.disabled_patterns = set(disabled_patterns) if disabled_patterns else set()
        self.enabled_patterns = set(enabled_patterns) if enabled_patterns else self.available_patterns
        self.similarity_threshold = similarity_threshold


        # Validate patterns
        unknown_disabled = self.disabled_patterns - self.available_patterns
        unknown_enabled = self.enabled_patterns - self.available_patterns
        if unknown_disabled or unknown_enabled:
            raise ValueError(
                f"Unknown patterns specified:\n" +
                (f"  Disabled: {unknown_disabled}\n" if unknown_disabled else "") +
                (f"  Enabled: {unknown_enabled}" if unknown_enabled else "")
            )

    def _extract_comments(self):
        """Extract comments and their line numbers from the source file."""
        source_lines = self.source.splitlines()
        for i, line in enumerate(source_lines, 1):
            line = line.strip()
            if line.startswith('#'):
                self.comments[i] = line
            elif '#' in line:
                self.comments[i] = line[line.index('#'):].strip()

    def _get_comment_block(self, start_line: int, end_line: int) -> List[str]:
        """Get all comments between two line numbers."""
        return [comment for line_no, comment in self.comments.items() 
                if start_line <= line_no <= end_line]

    def _analyze_print_statement(self, node: ast.Expr, function_name: str) -> PrintContext:
        """Analyze a print statement for relevance to a function."""
        if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name):
            return None

        print_text = self._extract_print_text(node)
        similarity = difflib.SequenceMatcher(None, print_text.lower(), 
                                           function_name.lower()).ratio()

        if self.debug:
            print(f"Analyzing print statement: '{print_text}'")
            print(f"Comparing to function: '{function_name}'")
            print(f"Similarity score: {similarity}")
            print(f"Current threshold: {self.similarity_threshold}")
            print(f"Result: {'MATCH' if similarity > self.similarity_threshold else 'NO MATCH'}")

        return PrintContext(
            node=node,
            text=print_text,
            similarity_to_function=similarity,
            line_number=node.lineno
        )

    def _should_include_print(self, print_context: PrintContext) -> bool:
        """Determine if a print statement should be included with its function."""
        return print_context.similarity_to_function > self.similarity_threshold

    def _extract_imports(self):
        """Extract all import statements from the source file."""
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.append(astor.to_source(node).strip())

    def _find_main_block(self) -> List[ast.stmt]:
        """Find the main execution block - either __main__ block or module-level code."""
        # First try to find explicit __main__ block
        for node in self.tree.body:
            if (isinstance(node, ast.If) and 
                isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == '__name__' and
                isinstance(node.test.comparators[0], ast.Constant) and
                node.test.comparators[0].value == '__main__'):
                return node.body
        
        # If no __main__ block found, collect module-level executable statements
        main_body = []
        for node in self.tree.body:
            # Skip imports, function/class definitions, and module docstrings
            if (not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, 
                                    ast.ClassDef, ast.AsyncFunctionDef)) and
                not (isinstance(node, ast.Expr) and 
                     isinstance(node.value, ast.Constant) and 
                     isinstance(node.value.value, str) and
                     node is self.tree.body[0])):  # Skip module docstring
                main_body.append(node)
        
        return main_body

    def get_available_patterns(self) -> Set[str]:
        return self.available_patterns

    def _is_setup_call(self, stmt: ast.stmt) -> bool:
        """Check if this statement matches any of our setup patterns."""
        try:
            return any(pattern(stmt) for pattern in FplitParser.setup_patterns.values())
        except AttributeError:
            return False

    def _extract_function_blocks(self, statements: List[ast.stmt]) -> List[Tuple[str, List[ast.stmt], List[str]]]:
        if hasattr(self, 'verbose') and self.verbose:
                print("\nActive patterns:", sorted(p for p in self.setup_patterns if self._is_pattern_active(p)))

        """Extract blocks of statements related to each function call."""
        function_blocks = []
        current_block = []
        current_function = None

        # Track the start line of the current block for comment extraction
        current_block_start = None
        
        for stmt in statements:
            # Debug pattern matching
            if hasattr(self, 'verbose') and self.verbose:
                print("\nChecking statement for setup patterns:", astor.to_source(stmt).strip())
                for pattern_name, pattern in self.setup_patterns.items():
                    if self._is_pattern_active(pattern_name):
                        try:
                            if pattern(stmt):
                                print(f"  Matched pattern: {pattern_name}")
                            else:
                                print(f"  Did not match: {pattern_name}")
                        except Exception as e:
                            print(f"  Error checking {pattern_name}: {e}")

            # Check if this is a function call
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if self._is_setup_call(stmt):
                    continue  # Skip this as it's a setup call
                if current_function:
                    block_comments = self._get_comment_block(current_block_start, stmt.lineno)
                    function_blocks.append((current_function, current_block, block_comments))
                current_block = [stmt]
                current_block_start = stmt.lineno
                current_function = astor.to_source(stmt.value.func).strip()
            # Check if this is a print statement
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and \
                 isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print':
                if current_function:
                    # Analyze print statement for relevance
                    print_context = self._analyze_print_statement(stmt, current_function)
                    if print_context and print_context.similarity_to_function > self.similarity_threshold:
                        current_block.append(stmt)
                if current_block:  # Only add print if we're in a block
                    current_block.append(stmt)
            # Check for assignment with function call
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                if current_function:
                    block_comments = self._get_comment_block(current_block_start, stmt.lineno)
                    function_blocks.append((current_function, current_block, block_comments))
                current_block = [stmt]
                current_block_start = stmt.lineno
                current_function = astor.to_source(stmt.value.func).strip()
            else:
                if current_block:  # Only add statements if we're in a block
                    current_block.append(stmt)

        # Don't forget the last block
        if current_function:
            block_comments = self._get_comment_block(current_block_start, stmt.lineno if stmt else float('inf'))
            function_blocks.append((current_function, current_block, block_comments))

        return function_blocks

    def _is_pattern_active(self, pattern_name):
        # If specific patterns are enabled, only use those
        if self.enabled_patterns != self.available_patterns:  # If user specified enables
            return pattern_name in self.enabled_patterns
        # Otherwise, use all patterns except explicitly disabled ones
        return pattern_name not in self.disabled_patterns

    def _extract_setup(self, disabled_patterns=None, enabled_patterns=None):
        """Extract module-level setup code that should be preserved."""
        active_patterns = {
            name: pattern 
            for name, pattern in self.setup_patterns.items() 
            if _is_pattern_active(name)
        }

        
        for node in self.tree.body:
            # Skip function/class definitions and imports
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                continue
                
            # Check if node matches any setup patterns
            for pattern_name, pattern in active_patterns.items():
                try:
                    if pattern(node):
                        if node.lineno in self.comments:
                            # Preserve any comment above the setup code
                            self.setup_nodes.append(ast.Expr(ast.Constant(self.comments[node.lineno])))
                        self.setup_nodes.append(node)
                        break
                except AttributeError:
                    # Skip any pattern matching errors due to unexpected node structure
                    continue

            # Also capture any standalone setup-related comments
            if (node.lineno in self.comments and
                any(keyword in self.comments[node.lineno].lower() 
                    for keyword in ['config', 'setup', 'initialize', 'settings'])):
                self.setup_nodes.append(ast.Expr(ast.Constant(self.comments[node.lineno])))

    def _extract_global_vars(self):
        """Extract global variable definitions and their values."""
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                # Only consider top-level assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.global_vars[target.id] = (node, set())
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                # Handle type-annotated assignments
                self.global_vars[node.target.id] = (node, set())

    def _track_global_usage(self, node: ast.AST, used_globals: Set[str]):
        """Recursively track which global variables are used in a node."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in self.global_vars:
                # Add this usage context to the set in our tuple
                self.global_vars[child.id][1].update(used_globals) 

    def _get_required_globals(self, statements: List[ast.stmt]) -> List[ast.AST]:
        """Get global variables needed by these statements."""
        used_globals = set()
        
        # Track usage in all statements
        for stmt in statements:
            self._track_global_usage(stmt, used_globals)
            
        # Return nodes for used globals in definition order
        return [self.global_vars[name][0] for name in self.global_vars 
                if any(usage in used_globals for usage in self.global_vars[name][1])]

    def _generate_file_content(self, function_name: str, statements: List[ast.stmt], 
                             comments: List[str], *, wrap_main: bool = False) -> str:
        """Generate content for a new file."""
        content = []
        
        # Add imports
        content.extend(self.imports)
        content.append("")  # Empty line after imports
        
        # Add required global variables
        globals_nodes = self._get_required_globals(statements)
        if globals_nodes:
            content.append("# Global variables")
            for node in globals_nodes:
                content.append(astor.to_source(node).strip())
            content.append("")  # Empty line after globals

        # Add relevant comments as documentation
        if comments:
            content.append("\"\"\"")
            for comment in comments:
                if comment.startswith('#'):
                    content.append(comment[1:].strip())
                else:
                    content.append(comment)
            content.append("\"\"\"")
            content.append("")
        


        # Add the main content, optionally wrapped
        if wrap_main:
            content.append("if __name__ == '__main__':")
            for stmt in statements:
                stmt_str = astor.to_source(stmt).strip()
                content.extend(f"    {line}" for line in stmt_str.split('\n'))
            content.append("    exit(0)")
        else:
            for stmt in statements:
                content.append(astor.to_source(stmt).strip())
            content.append("exit(0) if __name__ == '__main__' else None")
        
        return "\n".join(content)

    def split_into_files(self, output_dir: str = ".", verbose: bool = False, wrap_main: bool = False, no_setup: bool = False):
        """Split the main block into separate files."""
        self.verbose = verbose
        main_statements = self._find_main_block()
        if not main_statements:
            raise ValueError("No executable statements found in the source file")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        function_blocks = self._extract_function_blocks(main_statements)
        
        # Sort blocks by their starting line number to maintain original order
        function_blocks.sort(key=lambda x: x[1][0].lineno if x[1] else float('inf'))
        
        created_files = []
        for function_name, statements, comments in function_blocks:
            # Create safe filename from function name
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', function_name)
            filename = output_path / f"{safe_name}_demo.py"
            created_files.append(filename)
            
            if verbose:
                print(f"📝 Creating {filename.name}...")
            
            content = self._generate_file_content(function_name, statements, comments)
            
            with open(filename, 'w') as f:
                f.write(content)
        
        if verbose:
            print(f"\n📚 Created {len(created_files)} files:")
            for file in created_files:
                print(f"   - {file.name}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="fplit - Split Python modules into function-specific files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    fplit demo.py                 # Split into current directory
    fplit demo.py -o output_dir   # Split into specified directory
    fplit demo.py --verbose       # Show detailed splitting process
    fplit demo.py --wrap-main     # Force wrapping in __main__ blocks
        """
    )
    
    # File handling options
    parser.add_argument('source_file', help='Python source file to split')
    parser.add_argument('-o', '--output-dir', default='.',
                       help='Output directory for split files (default: current directory)')
    
    # Verbosity options
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase output verbosity (use -v or -vv)')
    
    # Analysis configuration
    parser.add_argument('--similarity-threshold', type=float, default=0.5,
                       help='Threshold for print statement similarity matching (0.0-1.0)')
    
    # Pattern loading options
    parser.add_argument('--patterns-dir', 
                       help='Directory containing custom pattern definitions')
    parser.add_argument('--skip-user-patterns', action='store_true',
                       help='Skip loading user pattern overrides')
    parser.add_argument('--skip-project-patterns', action='store_true',
                       help='Skip loading project-specific patterns')

    args = parser.parse_args()
    
    try:
        pattern_loader = PatternLoader(
            skip_user_patterns=args.skip_user_patterns,
            skip_project_patterns=args.skip_project_patterns,
            patterns_dir=args.patterns_dir
        )
        
        splitter = FplitParser(
            source_file=args.source_file,
            pattern_loader=pattern_loader,
            disabled_patterns=args.disable_patterns,
            enabled_patterns=args.enable_patterns,
            similarity_threshold=args.similarity_threshold
        )
        
        if args.list_patterns:
            print("Available setup patterns:")
            for pattern in splitter.get_available_patterns(): 
                print(f"  - {pattern}")
            exit(0)

        if args.verbose:
            print(f"🔍 Analyzing {args.source_file}...")
        
        splitter.split_into_files(args.output_dir, args.verbose, args.wrap_main, args.no_setup)
        
        if args.verbose:
            print("✨ Done! Your file has been ſplit into separate function demos.")        

    except FileNotFoundError as e:
        print(f"Error: Could not find file '{args.source_file}'")
        exit(1)
    except SyntaxError as e:
        print(f"Error: Invalid Python syntax in '{args.source_file}'")
        print(f"  Line {e.lineno}: {e.text}")
        print(f"  {' ' * (e.offset + 1)}^")  # Point to the error
        exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred:")
        print(f"  {type(e).__name__}: {str(e)}")
        if args.verbose:
            print("\nTraceback:")
            traceback.print_exc()
        exit(1)
