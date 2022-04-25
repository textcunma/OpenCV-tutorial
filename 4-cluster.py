import cv2
import argparse
import numpy as np

# 減色処理
def declease_color(input, img, args):

    # 繰り返し処理の終了条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 指定最大反復回数: 10、指定精度: 1.0
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER:
    # 「指定された精度に到達」もしくは「指定された繰り返し回数に到達」したら繰り返し計算を終了する

    # k-means(k平均法)
    """
    入力
    data : np.float32型かつ特徴ベクトルが一列なデータ
    K : クラスタ数（減色処理の場合は出力画像がK色になる）
    labels : すべてのサンプルのクラスタインデックスを格納する入力/出力整数配列
    criteria : 終了条件
    attempts : アルゴリズムが実行される回数
    flags : cv.KMEANS_RANDOM_CENTERS ランダムな初期クラスタ重心を設定

    出力
    labels : 各要素に与えられたラベルの配列
    centers : クラスタの重心の配列
    """
    _, label, center = cv2.kmeans(
        input, args.K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    center = np.uint8(center)  # 重心をuint8型に変形
    result = center[label.flatten()]
    result = result.reshape(img.shape)  # 幅と高さの情報を戻す

    return result


# キー色取得
def keycolor(img, args):
    img = np.asarray(img)
    img = img.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

    _, _, centers = cv2.kmeans(
        img, args.K, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )

    centers = centers.astype(np.uint8)  # float32 -> uint8
    return centers.tolist()  # Numpy配列をリストに変換


if __name__ == "__main__":
    # 設定
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="./img/cardiff.png", help="入力画像")
    parser.add_argument("--K", type=int, default=3, help="クラスタ数")
    args = parser.parse_args()

    # 入力画像を読み込み
    img = cv2.imread(args.image)  # shape:(597,687,3)

    # データ変形
    input = img.reshape(-1, 3)  # shape:(410139,3)  幅・高さの部分を2次元から1次元
    input = np.float32(input)  # cv2.kmeansに渡すデータはfloat型である必要があるため

    # 実行
    outputimg = declease_color(input, img, args)  # 減色処理
    color = keycolor(img, args)  # キー色処理

    # 表示
    # 減色処理結果
    h, w = img.shape[:2]  # 画像の高さと幅を取得
    img = cv2.resize(img, (int(w * 0.7), int(h * 0.7)))  # 表示のために変形
    outputimg = cv2.resize(outputimg, (int(w * 0.7), int(h * 0.7)))  # 表示のために変形
    displayimg = cv2.hconcat([img, outputimg])  # 画像を連結
    cv2.imshow("display", displayimg)  # 表示
    cv2.waitKey(0)

    # キー色結果
    img = np.zeros((200, 200 * args.K, 3), np.uint8)
    for i in range(args.K):
        print(color[i])
        cv2.rectangle(img, (200 * i, 0), (200 * (i + 1), 200), color[i], thickness=-1)
    cv2.imshow("display", img)
    cv2.waitKey(0)
