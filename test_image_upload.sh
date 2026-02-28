#!/bin/bash
# 测试图片上传API

# 创建一个简单的测试图片（1x1像素）
echo "创建测试图片..."
convert -size 100x100 xc:red /tmp/test_image.png 2>/dev/null || {
  # 如果没有imagemagick，使用Python创建
  python3 -c "
from PIL import Image
img = Image.new('RGB', (100, 100), color='red')
img.save('/tmp/test_image.png')
print('测试图片已创建: /tmp/test_image.png')
"
}

echo "测试图片上传API..."
curl -X POST http://localhost:8000/v1/generate-multipart \
  -F "prompt=这是什么颜色" \
  -F "image=@/tmp/test_image.png" \
  -F "max_new_tokens=50" \
  -F "temperature=0.7" \
  2>&1 | head -50

echo ""
echo "测试完成"
