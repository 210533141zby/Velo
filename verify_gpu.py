import docker
import sys

def check_gpu_access():
    try:
        client = docker.from_env()
    except Exception as e:
        print(f"❌ Docker client initialization failed: {e}")
        return False

    print("🔍 Testing Standard GPU Access (Runtime Hook)...")
    try:
        logs = client.containers.run(
            "ubuntu:22.04", 
            "nvidia-smi",
            remove=True,
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]
        )
        print("✅ Success! Standard NVIDIA Runtime is working.")
        print(logs.decode('utf-8'))
        return True
    except Exception as e:
        print(f"❌ Standard mode failed: {str(e).strip()}")

    print("\n🔍 Testing Privileged Mode (Workaround)...")
    try:
        # Check if we can see the device file at least
        logs = client.containers.run(
            "ubuntu:22.04", 
            "ls -l /dev/nvidia0",
            privileged=True,
            environment=["NVIDIA_VISIBLE_DEVICES=all"],
            remove=True
        )
        print("⚠️  Privileged Mode: Device /dev/nvidia0 is accessible.")
        print(f"   Output: {logs.decode('utf-8').strip()}")
        print("   Note: 'nvidia-smi' might still fail without mounted libs, but vLLM should work if it bundles libs or if we mount them.")
        return True
    except Exception as e:
        print(f"❌ Privileged mode failed: {str(e).strip()}")
        return False

if __name__ == "__main__":
    check_gpu_access()
