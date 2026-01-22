import runpod

def handler(event):
    """Simplest possible handler"""
    return {
        "status": "success",
        "message": "Hello from RunPod!",
        "received": event.get("input", {})
    }

runpod.serverless.start({"handler": handler})