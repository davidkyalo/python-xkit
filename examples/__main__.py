import os
import sys

import examples
from zana.util import try_import

print(sys.argv)

path = sys.argv[-1] if len(sys.argv) > 1 else None

if path:
    try_import(f"examples.{path}.run")()
