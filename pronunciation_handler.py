import runpod
import os
import sys

print("🚀 Starting pronunciation handler...")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")
    sys.exit(1)

try:
    import torchaudio
    print(f"✅ TorchAudio imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TorchAudio: {e}")
    sys.exit(1)

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    print(f"✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Transformers: {e}")
    sys.exit(1)

# Try to load the model
MODEL_NAME = "gigant/romanian-wav2vec2"
print(f"📥 Loading model: {MODEL_NAME}")

try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    print("✅ Processor loaded")
    
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ Model loaded and ready!")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def handler(event):
    """
    Main handler for RunPod serverless
    """
    try:
        print("📨 Received request")
        input_data = event.get("input", {})
        
        # For now, just return a simple success message
        # We'll add actual audio processing later
        return {
            "status": "success",
            "message": "Pronunciation model is running!",
            "model": MODEL_NAME,
            "received_input": str(input_data.keys())
        }
        
    except Exception as e:
        print(f"❌ Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

print("✅ Handler ready, starting RunPod serverless...")
runpod.serverless.start({"handler": handler})