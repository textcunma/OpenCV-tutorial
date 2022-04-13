# 参考：https://github.com/hpc203/Yolo-Fastest-opencv-dnn/blob/master/main_yolov3.py
# 参考：https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
import cv2
import argparse
import numpy as np

# パラメータ
confThreshold = 0.25    # 信頼度スコア閾値
nmsThreshold = 0.4      # NMS閾値

# ニューラルネットワークの設定と重み
modelConfiguration = "./Yolo-Fastest-opencv-dnn/Yolo-Fastest-voc/yolo-fastest-xl.cfg"
modelWeights = "./Yolo-Fastest-opencv-dnn/Yolo-Fastest-voc/yolo-fastest-xl.weights"

# クラス名読み込み
classesFile = "./Yolo-Fastest-opencv-dnn/voc.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]

# 出力レイヤ名取得
def getOutputsNames(net):
    layersNames = net.getLayerNames()   # 全レイヤの名前を取得
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]  # 出力レイヤの名前を取得

# 予測バウンディングボックス描画
def drawPred(classId, conf, left, top, right, bottom):
    cv2.rectangle(img, (left, top), (right, bottom), (0,0,255), thickness=4)   # バウンディングボックス描画
    label = '%.2f' % conf   # 信頼度スコア[%]　ラベル

    # ラベル | クラス名 | 信頼度スコア　を取得
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # バウンディングボックス上部にラベルを表示
    labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(img, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    print(label)

# NMSを用いた後処理
def postprocess(img, outs):
    '''
    img : 入力画像
    outs : 予測出力
    '''
    # 予測されたBBoxから信頼度スコアの高いものだけを残す
    h,w=img.shape[:2]           # 高さ幅取得
    classIds = []               # 物体クラスIDリスト
    confidences = []            # 信頼度スコアリスト
    boxes = []                  # BBoxリスト

    '''
    物体クラス = [犬、猫、鳥、自転車]の4クラスとすると
    信頼度スコアが[0.95,0.1,0.4,0.00] とした場合、スコアが最大な要素の添え字は「0」、つまり犬と予測
    '''

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)             # 各種物体クラスの中で信頼度スコアが最大要素の添え字(ID)を取得  # ex) classId=0
            confidence = scores[classId]            # 信頼度スコアを取得                                        # ex) confidence=0.95
            if confidence > confThreshold:          # 信頼度スコアが閾値より大きいならば...
                center_x = int(detection[0] * w)    # BBoxの中心X
                center_y = int(detection[1] * h)    # BBoxの中心Y
                width = int(detection[2] * w)       # BBoxの幅
                height = int(detection[3] * h)      # BBoxの高さ
                left = int(center_x - width / 2)    # 中心Xと幅から左上点の座標Xを算出
                top = int(center_y - height / 2)    # 中心Yと高さから左上点の座標Yを算出
                classIds.append(classId)                # 物体クラスIDをリストに記録
                confidences.append(float(confidence))   # 信頼度スコアをリストに記録
                boxes.append([left, top, width, height])    # 左上点の座標(X,Y)、幅、高さをリストに記録

    # NMSを用いて信頼度スコアが低いものを削除
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # 描画処理
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # 左上点と右下点の座標を用いてBBox描画 | クラス名と信頼度スコアも描画  
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)  

if __name__=='__main__':
    # 設定
    parser = argparse.ArgumentParser(description='YOLOv3を用いた物体検出')
    parser.add_argument('--image', type=str, default="./img/face.png", help='入力画像パス指定')  ##############################
    args = parser.parse_args()

    # ニューラルネットワーク設定
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)  # Darknetからニューラルネットを読み込む
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)                     # GPU版もあるらしいです
    
    # 入力
    img = cv2.imread(args.image)

    # 入力画像を4次元blob形式に変換
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)
        # img : 入力画像
        # 1/255.0 : スケール係数：ピクセル値をスケーリング：「1/255」ピクセル値が間隔[0,1]に正規化  # 参考：https://note.com/navitime_tech/n/nae344375d0c9
        # (320,320) : ニューラルネットに入力する画像サイズ
        # swapRB=True : BGR -> RGB
        # crpo=False : 画像をトリミングしない（アスペクト比そのまま）

    # ニューラルネットに画像を入力
    net.setInput(blob)

    # ニューラルネットを動作、出力を得る
    outs = net.forward(getOutputsNames(net))
 
    # 信頼度スコアが低いBBoxを削除する後処理
    postprocess(img, outs)

    # 表示
    cv2.imshow("display", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()