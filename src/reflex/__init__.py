def main() -> None:
    """Console-script entry point; imports lazily so importing the package stays cheap."""
    from reflex.main import main as run_demo

    run_demo()
