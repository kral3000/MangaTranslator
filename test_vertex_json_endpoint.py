from pathlib import Path
import importlib.util
import sys

root = Path(__file__).resolve().parent
json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "keys" / "service-account.json"

if not json_path.exists():
    raise SystemExit(f"Missing JSON file: {json_path}")

json_text = json_path.read_text(encoding="utf-8")

endpoint_path = root / "utils" / "endpoints" / "google_vertex_json_final.py"
spec = importlib.util.spec_from_file_location("google_vertex_json_final_test", endpoint_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print(module.call_gemini_endpoint(
    json_text,
    "gemini-2.5-flash",
    [{"text": "한국어로 OK만 출력해줘"}],
    {"temperature": 0.1, "maxOutputTokens": 20},
    debug=True,
))
