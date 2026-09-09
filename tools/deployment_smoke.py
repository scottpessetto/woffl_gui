"""Offline deployment smoke check; imports the app without starting its lifespan."""
from importlib import metadata
from pathlib import Path
import sys
import sysconfig


def main():
    import woffl
    from server.main import app
    from woffl.flow.entry_energy import MODEL_VERSION

    root = Path(__file__).resolve().parents[1]
    assert sys.version_info >= (3, 11), "App dependencies require Python >=3.11"
    assert Path(woffl.__file__).resolve() == root / "woffl" / "__init__.py"
    # A local egg-info directory is fine; a second site-packages copy is not.
    sites = {sysconfig.get_path(key) for key in ("purelib", "platlib")}
    installed = {d.metadata["Name"].lower() for d in metadata.distributions(path=sites)}
    assert "woffl" not in installed, "Remove the PyPI woffl copy from this environment"
    assert app.routes
    print(f"API imports with local physics {MODEL_VERSION}; no PyPI woffl installed.")


if __name__ == "__main__":
    main()
