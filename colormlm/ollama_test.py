# -*- coding: utf-8 -*-
"""Ollama 7B 基线速度测试"""
import time
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt, model="qwen2.5:7b", max_tokens=100):
    """Send prompt to ollama and measure speed"""
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }).encode("utf-8")
    
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start
        
        response_text = result.get("response", "")
        eval_count = result.get("eval_count", 0)
        eval_duration_ns = result.get("eval_duration", 1)
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0
        
        return {
            "response": response_text,
            "total_time": elapsed,
            "tokens": eval_count,
            "tokens_per_sec": tokens_per_sec,
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

# 测试用例
test_prompts = [
    {
        "name": "Speed Test (short)",
        "prompt": "Hello, how are you?",
        "max_tokens": 30
    },
    {
        "name": "Code Generation",
        "prompt": "Write a Python function to find fibonacci numbers:",
        "max_tokens": 100
    },
    {
        "name": "Code Understanding",
        "prompt": "Explain what this code does:\ndef quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
        "max_tokens": 80
    },
    {
        "name": "Long Generation",
        "prompt": "Write a Python class for a binary search tree with insert, search, and delete methods:",
        "max_tokens": 200
    }
]

print("="*60)
print("Ollama 7B Baseline Speed Test")
print("="*60)

# Check if model is available
print("\nChecking model availability...")
test = query_ollama("Hi", max_tokens=5)
if not test["success"]:
    print(f"Error: {test['error']}")
    print("Make sure ollama is running and model is downloaded:")
    print("  ollama pull qwen2.5:7b")
    exit(1)

print(f"Model ready! First response in {test['total_time']:.1f}s\n")

results = []
for tp in test_prompts:
    print(f"Testing: {tp['name']}...")
    r = query_ollama(tp["prompt"], max_tokens=tp["max_tokens"])
    if r["success"]:
        results.append(r)
        print(f"  Time: {r['total_time']:.1f}s | Tokens: {r['tokens']} | Speed: {r['tokens_per_sec']:.1f} tok/s")
        print(f"  Response: {r['response'][:150]}...")
    else:
        print(f"  Error: {r['error']}")
    print()

# Summary
print("="*60)
print("Baseline Summary")
print("="*60)
if results:
    avg_speed = sum(r["tokens_per_sec"] for r in results) / len(results)
    print(f"Average speed: {avg_speed:.1f} tokens/sec")
    print(f"Model: qwen2.5:7b (4-bit quantized)")
    print(f"This is the baseline to beat with FSQ compression.")
