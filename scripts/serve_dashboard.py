"""Start the local FailureAtlas dashboard."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("failure_atlas.web:app", host="127.0.0.1", port=8000, reload=False)
