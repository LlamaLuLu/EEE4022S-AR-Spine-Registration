"""Import this first in every test file: puts scripts/registration on the path."""
import sys
from pathlib import Path

REG = Path(__file__).resolve().parents[1] / "scripts" / "registration"
if str(REG) not in sys.path:
    sys.path.insert(0, str(REG))