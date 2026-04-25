"""Run the current Flask server entrypoint."""

from src.jsjb.core.config import load_runtime_config
from src.jsjb.web.app import create_app

app = create_app()

if __name__ == "__main__":
    config = load_runtime_config()
    app.run(host=config.host, port=config.enhanced_port, debug=config.debug)
