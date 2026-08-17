from PIL import Image

content_path = 'image/content2.jpg'
output_path = 'output/output.jpg'

# 获取原始内容图尺寸
orig_size = Image.open(content_path).size
print(f"原始尺寸: {orig_size[0]} × {orig_size[1]}")

# 打开 output.jpg 并缩放到原始尺寸
img = Image.open(output_path)
print(f"当前尺寸: {img.size[0]} × {img.size[1]}")

img = img.resize(orig_size, Image.LANCZOS)
img.save(output_path)
print(f"已恢复为原始尺寸并保存: {output_path}")