import requests

url = "https://huggingface.co/babakhani/hwnet-v2/resolve/main/hwnetv2.pth"
r = requests.get(url, stream=True)

with open("hwnetv2.pth", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("✅ HWNet-v2 weights downloaded successfully")
