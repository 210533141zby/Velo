import urllib.request
import os

url = "https://cdn.staticfile.org/font-awesome/6.4.0/css/all.min.css"
dest_dir = r"d:\Working\seek\Wiki\backend\static\vendor"
dest_file = os.path.join(dest_dir, "font-awesome.min.css")

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

print(f"Downloading {url} to {dest_file}...")
try:
    with urllib.request.urlopen(url) as response, open(dest_file, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("Download success!")
except Exception as e:
    print(f"Download failed: {e}")
