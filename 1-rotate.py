"""
<画像処理ライブラリ>
・Pillow(PIL:Python Imaging Library)
・Numpy
・OpenCV
"""

from PIL import Image  # Pillow(PIL)
import numpy as np  # Numpy
import cv2  # OpenCV

display_flg = True  # 画像表示の有無フラグ

### PIL | Numpy　を用いた処理
img = Image.open("img/cardiff.png")  # 画像読み込み

## PIL
rotate_img_PIL = img.rotate(-90)  # PILライブラリを用いた回転
if display_flg:
    rotate_img_PIL.show()  # 画像表示

## Numpy
img = np.array(img)  # Numpy配列に変換
print("画像データ", img)  # 画像データの中身(0~255)
print("Numpyデータ次元", img.shape)  # (597, 687, 4) (高さ,幅,チャンネル数)  ※チャンネル：R,G,B,α
rotate_img_Numpy = np.rot90(img)  # Numpyライブラリを用いた回転
if display_flg:
    show_img_PIL = Image.fromarray(rotate_img_Numpy)  # 表示のためにPILに変換
    show_img_PIL.show()  # 画像表示


### OpenCVを用いた処理
img = cv2.imread("img/cardiff.png")  # 画像読み込み
h, w = img.shape[:2]  # 高さ、幅取得
print("OpenCVデータ次元", img.shape)  # (597, 687, 3) (高さ,幅,チャンネル数)  ※チャンネル：R,G,B

alpha_img = cv2.imread("img/cardiff.png", -1)  # αチャンネルを含む画像読み込み
print("OpenCVデータ次元 α入り", alpha_img.shape)  # (597, 687, 4) (高さ,幅,チャンネル数)  ※チャンネル：R,G,B,α

# パターン1
rotate_img_CV = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 時計回りに90度回転
if display_flg:
    cv2.imshow("rotation", rotate_img_CV)  # 画像表示
    cv2.waitKey(0)  # imshowとセットで書かないと表示されない

# パターン2
centerX = w // 2  # 中心座標 X
centerY = h // 2  # 中心座標 Y
ratio = 1.0  # 拡大縮小比率
degree = -90  # 回転角度

matrix = cv2.getRotationMatrix2D(
    (centerX, centerY), degree, ratio
)  # 回転、拡大縮小に関する変換行列生成(中心座標,回転角度,拡大縮小比率)
affine_img = cv2.warpAffine(img, matrix, (w, h))  # アフィン変換
if display_flg:
    cv2.imshow("rotation", affine_img)  # 画像表示
    cv2.waitKey(0)  # imshowとセットで書かないと表示されない

# getRotationMatrix2D 内部処理
rad = np.deg2rad(-degree)  # 度数 -> ラジアン
sin = np.float32(np.sin(rad))  # sin値算出
cos = np.float32(np.cos(rad))  # cos値算出
trans_matrix = np.float32(
    [[1, 0, centerX], [0, 1, centerY], [0, 0, 1]]
)  # 平行移動行列(原点へ移動)
rot_matrix = np.float32([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])  # 回転行列
re_trans_matrix = np.float32(
    [[1, 0, -centerX], [0, 1, -centerY], [0, 0, 1]]
)  # 平行移動行列(元へ戻す)
size_matrix = np.float32([[1 * ratio, 0, 0], [0, 1 * ratio, 0], [0, 0, 1]])  # 拡大縮小行列
tmp = np.dot(trans_matrix, rot_matrix)  # 平行移動行列と回転行列を合成
tmp = np.dot(tmp, size_matrix)  # 変換行列と平行移動行列を合成
matrix = np.dot(tmp, re_trans_matrix)  # 変換行列と拡大縮小行列を合成
print("変換行列")
print(matrix)
matrix2 = matrix[:2]  # warpAffine関数はfloat32かつ2*3行列しか対応しないため変換
affine_img = cv2.warpAffine(img, matrix2, (w, h))  # アフィン変換
if display_flg:
    cv2.imshow("rotation", affine_img)  # 画像表示
    cv2.waitKey(0)  # imshowとセットで書かないと表示されない
