# 参考：https://github.com/zihaozhang9/ENet/blob/58995f1325bf5a3747e6c185a3114c0a576786fd/segment.py

import numpy as np
import argparse
import cv2

def model(img,args):
    net = cv2.dnn.readNet(args.model)       # モデル読み込み

    # ENetは1024×512の画像で学習させているため、そのサイズに変形
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (1024, 512), swapRB=True, crop=False)
    
    net.setInput(blob)      # blobをニューラルネットにセット
    output = net.forward()  # ニューラルネット動作
    return output

def draw(img,output,color,args):
    """
    output :モデル出力  shape:(1,クラス数,高さ,幅)
    img : 入力画像
    color :各クラスに対応した色情報 
    """
    classMap = np.argmax(output[0], axis=0) # クラスIDマップ
    colormask = color[classMap]  # クラスIDマップとそのクラスに対応した色マップを作製
    
    # 入力画像の高さと幅になるようにリサイズ
    colormask = cv2.resize(colormask, (img.shape[1], img.shape[0]),interpolation=cv2.INTER_NEAREST)

    # 出力画像を生成
    if args.colordisplay:
        outputimg = ((0.4 * img) + (0.6 * colormask)).astype("uint8")   #入力画像とカラーマスクを合成
    else:
        outputimg=colormask # カラーマスクをそのまま出力画像にする

    return outputimg


if __name__=='__main__':
    # 設定
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default="./ENet/enet-cityscapes/enet-model.net",help="モデルパス")
    parser.add_argument("--classes", default="./ENet/enet-cityscapes/enet-classes.txt",help="クラスラベルパス")
    parser.add_argument("--image", default="./ENet/images/example_01.png",help="入力画像")
    parser.add_argument("--colors", type=str,default="./ENet/enet-cityscapes/enet-colors.txt",help="各クラスに対応した色パス")
    parser.add_argument("--colordisplay",  action='store_false',help="出力画像を合成かカラーマスク画像か")  # デフォルト：True
    args = parser.parse_args()

    # 入力画像を読み込み
    img=cv2.imread(args.image)

    # クラスラベル名を取得（'road','sidewalk'など）
    classname = open(args.classes).read().strip().split("\n")

    # 各クラスに対応した色の情報を取得
    color = open(args.colors).read().strip().split("\n")
    color = [np.array(c.split(",")).astype("int") for c in color]
    color = np.array(color, dtype="uint8")    

    # モデル実行
    output=model(img,args)

    # 出力画像生成
    outputimg=draw(img,output,color,args)

    # 表示
    h,w=img.shape[:2]                                           # 画像の高さと幅を取得
    img=cv2.resize(img , (int(w*0.3), int(h*0.3)))              # 表示のために変形
    outputimg=cv2.resize(outputimg , (int(w*0.3), int(h*0.3)))  # 表示のために変形
    displayimg = cv2.vconcat([img, outputimg])                  # 画像を連結
    cv2.imshow("display", displayimg)                           # 表示
    cv2.waitKey(0)