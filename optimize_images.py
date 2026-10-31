from pathlib import Path
from PIL import Image

ROOT = Path("results")   # chỉ quét thư mục results
DELETE_ORIGINAL_PNG = False   # đổi thành True nếu muốn xóa PNG sau khi convert thành công

total_old = 0
total_new = 0
count = 0

for png_path in ROOT.rglob("*.png"):
    webp_path = png_path.with_suffix(".webp")

    try:
        with Image.open(png_path) as img:
            img.save(
                webp_path,
                format="WEBP",
                lossless=True,   # giữ chất lượng, không vỡ ảnh
                method=6         # nén tốt hơn
            )

        old_size = png_path.stat().st_size
        new_size = webp_path.stat().st_size

        total_old += old_size
        total_new += new_size
        count += 1

        print(f"[OK] {png_path} -> {webp_path}")
        print(f"     {old_size/1024:.1f} KB -> {new_size/1024:.1f} KB")

        if DELETE_ORIGINAL_PNG and webp_path.exists():
            png_path.unlink()
            print("     Deleted original PNG")

    except Exception as e:
        print(f"[ERROR] {png_path}: {e}")

print("\n====== SUMMARY ======")
print(f"Converted files : {count}")
print(f"Old total size  : {total_old/1024:.1f} KB")
print(f"New total size  : {total_new/1024:.1f} KB")

if total_old > 0:
    saved = total_old - total_new
    print(f"Saved           : {saved/1024:.1f} KB ({saved/total_old*100:.2f}%)")