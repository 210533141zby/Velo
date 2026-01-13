import urllib.request
import os

# Font Awesome 6.4.0 base URL
base_url = "https://cdn.staticfile.org/font-awesome/6.4.0/webfonts/"
dest_dir = r"d:\Working\seek\Wiki\backend\static\webfonts"

# List of fonts to download (based on standard FA distribution)
fonts = [
    "fa-brands-400.woff2",
    "fa-brands-400.ttf",
    "fa-regular-400.woff2",
    "fa-regular-400.ttf",
    "fa-solid-900.woff2",
    "fa-solid-900.ttf",
    "fa-v4compatibility.woff2",
    "fa-v4compatibility.ttf"
]

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for font in fonts:
    url = base_url + font
    dest_file = os.path.join(dest_dir, font)
    print(f"Downloading {font}...")
    try:
        with urllib.request.urlopen(url) as response, open(dest_file, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Success: {font}")
    except Exception as e:
        print(f"Failed to download {font}: {e}")
