"""Exercise the real dashboard incident renderer with every stored evidence shape."""
import html
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='dashboard renderer requires Node.js')

NODE = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {
  esc: (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
  kv: () => '', row: (r) => r, input,
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('const NAMES'), source.indexOf('function commitRow')) + '\nresult = input.map(incident);', context);
process.stdout.write(JSON.stringify(context.result));
'''


def render(evidence):
    record = {'id': 'CASE-1', 'title': 'test', 'area': 'test', 'impact': 'test', 'cause': 'test',
              'attribution': 'test', 'evidence': evidence, 'events': []}
    result = subprocess.run(['node', '-e', NODE, str(ROOT / 'research-bug-board/app.js')],
                            input=json.dumps([record]), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)[0]['body']


def test_user_report_is_rendered_as_text():
    body = render([{'kind':'user_report', 'text':'MODEL CHAIN WAIT: node=example\nheartbeat_age=12s'}])
    assert 'MODEL CHAIN WAIT: node=example' in body
    assert 'heartbeat_age=12s' in body
    assert 'undefined' not in body
    assert 'href="#"' not in body


def test_source_commit_path_links_to_the_real_file():
    sha = 'a' * 40
    body = render([{'kind':'source', 'commit':sha, 'path':'srgc_research/dispatch/model_chain.py'}])
    assert f'https://github.com/33modeling/offpolicy-misranking/blob/{sha}/srgc_research/dispatch/model_chain.py' in body
    assert 'srgc_research/dispatch/model_chain.py' in body
    assert 'undefined' not in body


def test_legacy_evidence_links_keep_their_labels_and_urls():
    url = "https://github.com/33modeling/offpolicy-misranking/commit/1234567"
    body = render([{"kind": "commit", "label": "original fix", "url": url}])
    assert f'href="{url}"' in body
    assert "original fix" in body


def test_report_text_is_escaped_and_newlines_are_retained():
    body = render([{"kind": "user_report", "text": '<script>alert("x")</script>\nnext line'}])
    assert "<script>" not in body
    assert "&lt;script&gt;" in body
    assert "<br>next line" in body


@pytest.mark.parametrize("commit,path", [("not-a-sha", "file.py"), ("a" * 40, "../file.py"),
                                        ("a" * 40, "/file.py"), ("a" * 40, "x//file.py")])
def test_invalid_source_identity_does_not_manufacture_a_link(commit, path):
    body = render([{"kind": "source", "commit": commit, "path": path}])
    assert "<a " not in body
    assert "href=" not in body


def test_all_stored_report_text_and_structured_source_links_are_visible():
    records = json.loads((ROOT / "research-bug-board/incidents.json").read_text())
    for record in records:
        evidence = record.get("evidence", [])
        body = render(evidence)
        assert 'href="#"' not in body, record["id"]
        for item in evidence:
            if item.get("kind") == "user_report":
                assert html.escape(item["text"], quote=True).replace("&#x27;", "&#39;").replace("\n", "<br>") in body
            elif item.get("kind") == "source" and "commit" in item and "path" in item:
                assert f'/blob/{item["commit"]}/{item["path"]}' in body
