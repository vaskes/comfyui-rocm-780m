#!/usr/bin/env python3
import json, sys
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/history", timeout=5) as r:
    d = json.load(r)
print(f"Total prompts: {len(d)}")
for pid, h in d.items():
    outputs = h.get('outputs', {})
    status = h.get('status', '?')
    imgs = sum(len(o.get('images', [])) for o in outputs.values())
    print(f"  {pid[:12]} status={status}  imgs={imgs}  outputs={list(outputs.keys())}")
