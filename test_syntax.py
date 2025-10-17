#!/usr/bin/env python3
import ast
import sys

try:
    with open('main.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("✅ main.py 语法检查通过!")
    sys.exit(0)
except SyntaxError as e:
    print(f"❌ 语法错误: {e}")
    print(f"  行号: {e.lineno}")
    print(f"  位置: {e.offset}")
    print(f"  文本: {e.text}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    sys.exit(1)
