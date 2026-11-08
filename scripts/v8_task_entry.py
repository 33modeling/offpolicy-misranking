"""Run the exact original v8 child with a separately receipted decode observer."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

from v8_generation_probe import install


def main():
    path=Path(sys.argv[1]);expected=os.environ.get('V8_ORIGINAL_CHILD_SHA256')
    if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Original v8 child bytes differ from the selected runner identity')
    spec=importlib.util.spec_from_file_location('_v8_original_child',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    install(int(os.environ.get('RANK','0')))
    sys.argv=[str(path),*sys.argv[2:]]
    return module.main()


if __name__=='__main__':raise SystemExit(main())
