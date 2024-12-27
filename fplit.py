import ast
import astor
from pathlib import Path
import re
from typing import List, Tuple, Optional

class MainBlockSplitter:
    def __init__(self, source_file: str):
        self.source_file = Path(source_file)
        with open(source_file, 'r') as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)
        self.imports = []
        self._extract_imports()

    def _extract_imports(self):
        """Extract all import statements from the source file."""
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.append(astor.to_source(node).strip())

    def _find_main_block(self) -> Optional[ast.If]:
        """Find the if __name__ == '__main__' block."""
        for node in self.tree.body:
            if (isinstance(node, ast.If) and 
                isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == '__name__' and
                isinstance(node.test.comparators[0], ast.Constant) and
                node.test.comparators[0].value == '__main__'):
                return node
        return None

    def _extract_function_blocks(self, main_block: ast.If) -> List[Tuple[str, List[ast.stmt]]]:
        """Extract blocks of statements related to each function call."""
        function_blocks = []
        current_block = []
        current_function = None

        for stmt in main_block.body:
            # Check if this is a function call
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if current_function:
                    function_blocks.append((current_function, current_block))
                current_block = [stmt]
                current_function = astor.to_source(stmt.value.func).strip()
            # Check if this is a print statement
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and \
                 isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print':
                if current_block:  # Only add print if we're in a block
                    current_block.append(stmt)
            # Check for assignment with function call
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                if current_function:
                    function_blocks.append((current_function, current_block))
                current_block = [stmt]
                current_function = astor.to_source(stmt.value.func).strip()
            else:
                if current_block:  # Only add statements if we're in a block
                    current_block.append(stmt)

        if current_function:  # Don't forget the last block
            function_blocks.append((current_function, current_block))

        return function_blocks

    def _generate_file_content(self, function_name: str, statements: List[ast.stmt]) -> str:
        """Generate content for a new file."""
        content = []
        
        # Add imports
        content.extend(self.imports)
        content.append("")  # Empty line after imports
        
        # Add main block
        content.append("if __name__ == '__main__':")
        for stmt in statements:
            # Indent the statement
            stmt_str = astor.to_source(stmt).strip()
            content.extend(f"    {line}" for line in stmt_str.split('\n'))
        
        content.append("    exit(0)")
        return "\n".join(content)

    def split_into_files(self, output_dir: str = "."):
        """Split the main block into separate files."""
        main_block = self._find_main_block()
        if not main_block:
            raise ValueError("No __main__ block found in the source file")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        function_blocks = self._extract_function_blocks(main_block)
        
        for function_name, statements in function_blocks:
            # Create safe filename from function name
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', function_name)
            filename = output_path / f"{safe_name}_demo.py"
            
            content = self._generate_file_content(function_name, statements)
            
            with open(filename, 'w') as f:
                f.write(content)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <source_file>")
        exit(1)
    
    fplit = MainBlockSplitter(sys.argv[1])
    fplit.split_into_files()
