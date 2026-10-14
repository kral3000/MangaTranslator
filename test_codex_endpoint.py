import base64
import importlib.util
import json
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "codex_endpoint_test", Path(__file__).parent / "utils/endpoints/codex.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

# Notifications may precede the turn/start reply; concurrent turns finish out of order.
_PEER = r"""
import json, os, sys, threading, time
lock = threading.Lock()
threads = {}
def send(message):
    with lock:
        print(json.dumps(message), flush=True)
def finish(request):
    p = request['params']; tid = p['threadId']; text = p['input'][0]['text']
    if text == 'CRASH': os._exit(1)
    def event(method, **params):
        send({'method':method,'params':{'threadId':tid,**params}})
    event('turn/started', turn={'id':tid})
    event('item/completed', item={'type':'agentMessage','phase':'commentary','text':'ignore this'})
    if text == 'START_HANG': return
    send({'id':request['id'],'result':{'turn':{'id':tid}}})
    if text == 'HANG': return
    time.sleep(.04 if text == 'slow' else .005)
    result = json.dumps({'input':p['input'],'thread':tid,'settings':threads[tid],'effort':p.get('effort')})
    event('item/completed', item={'type':'agentMessage','phase':'final_answer','text':'' if text == 'EMPTY' else result})
    event('thread/tokenUsage/updated', tokenUsage={'last':{'inputTokens':10,'outputTokens':3,'reasoningOutputTokens':1,'totalTokens':13}})
    event('turn/completed', turn={'id':tid,'status':'failed' if text == 'FAIL' else 'completed','error':{'message':'quota exceeded'} if text == 'FAIL' else None})
for line in sys.stdin:
    r = json.loads(line); method = r.get('method'); result = {}
    if 'id' not in r: continue
    if method == 'config/read': result = {'config':{'mcp_servers':{'example':{}}}}
    elif method == 'skills/list': result = {'data':[{'skills':[{'path':'fake/SKILL.md'}]}]}
    elif method == 'model/list':
        cursor = r['params'].get('cursor')
        result = {'data':[{'model':'new-model' if not cursor else 'next-model','supportedReasoningEfforts':[{'reasoningEffort':'low'},{'reasoningEffort':'high'}]}],'nextCursor':None if cursor else 'page2'}
    elif method == 'thread/start':
        tid = str(len(threads)+1); threads[tid] = r['params']; result = {'thread':{'id':tid}}
    elif method == 'turn/start':
        threading.Thread(target=finish,args=(r,),daemon=True).start(); continue
    elif method == 'turn/interrupt':
        send({'method':'turn/completed','params':{'threadId':r['params']['threadId'],'turn':{'status':'interrupted'}}})
    send({'id':r['id'],'result':result})
"""


class CodexEndpointTest(unittest.TestCase):
    def setUp(self):
        _MODULE.close_codex_connection()
        _MODULE._catalog = []
        self.command = patch.object(
            _MODULE, "_codex_command", return_value=[sys.executable, "-u", "-c", _PEER]
        )
        self.command.start()

    def tearDown(self):
        _MODULE.close_codex_connection()
        _MODULE._catalog = []
        self.command.stop()

    def test_reuses_stdio_and_preserves_order_and_concurrent_responses(self):
        self.assertEqual(
            [m["model"] for m in _MODULE.codex_catalog(refresh=True)],
            ["new-model", "next-model"],
        )
        image = base64.b64encode(b"image").decode()
        parts = [
            {"text": "slow"},
            {"inline_data": {"mime_type": "image/png", "data": image}},
            {"text": "item 1"},
        ]
        usage = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                _MODULE.call_codex_endpoint,
                "new-model",
                parts,
                "translate only",
                usage_output=usage,
            )
            second = pool.submit(
                _MODULE.call_codex_endpoint, "new-model", [{"text": "fast"}]
            )
            fast = json.loads(second.result())
            slow = json.loads(first.result())
        server = _MODULE._server
        self.assertNotEqual(slow["thread"], fast["thread"])
        self.assertEqual([p["type"] for p in slow["input"]], ["text", "image", "text"])
        self.assertEqual(slow["input"][1]["url"], f"data:image/png;base64,{image}")
        self.assertEqual(slow["settings"]["developerInstructions"], "translate only")
        self.assertTrue(slow["settings"]["ephemeral"])
        self.assertFalse(slow["settings"]["config"]["mcp_servers.example.enabled"])
        self.assertEqual(
            slow["settings"]["config"]["skills.config"],
            [{"path": "fake/SKILL.md", "enabled": False}],
        )
        self.assertEqual(slow["effort"], "low")
        self.assertEqual(
            usage, {"prompt": 10, "thoughts": 1, "output": 2, "total": 13, "calls": 1}
        )
        self.assertFalse(server._events)
        again = json.loads(
            _MODULE.call_codex_endpoint("new-model", [{"text": "again"}])
        )
        self.assertIs(server, _MODULE._server)
        self.assertNotEqual(again["thread"], slow["thread"])

    def test_timeout_failure_empty_response_and_process_recovery(self):
        _MODULE.codex_catalog(refresh=True)
        for text, expected in [
            ("START_HANG", "timed out"),
            ("HANG", "timed out"),
            ("FAIL", "quota exceeded"),
            ("EMPTY", "empty response"),
            ("CRASH", "process stopped"),
        ]:
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(_MODULE.TranslationError, expected),
            ):
                _MODULE.call_codex_endpoint("new-model", [{"text": text}], timeout=0.15)
        result = json.loads(
            _MODULE.call_codex_endpoint("new-model", [{"text": "recovered"}])
        )
        self.assertEqual(result["input"][0]["text"], "recovered")

    def test_rejects_invalid_images_and_unsupported_effort(self):
        with self.assertRaises(_MODULE.ValidationError):
            _MODULE.call_codex_endpoint(
                "new-model",
                [{"text": "translate"}, {"inline_data": {"data": "broken"}}],
            )
        self.assertIsNone(_MODULE._server)
        _MODULE.codex_catalog(refresh=True)
        with self.assertRaises(_MODULE.ValidationError):
            _MODULE.call_codex_endpoint(
                "new-model", [{"text": "translate"}], reasoning_effort="ultra"
            )


if __name__ == "__main__":
    unittest.main()
