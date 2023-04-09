import cv2 
import numpy as np

net = cv2.dnn.readNetFromDarknet('darknet/cfg/yolov4.cfg', 'darknet/yolov4.weights')

# Carrega as classes de objetos
classes = []
with open('darknet/data/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Carrega a imagem
img = cv2.imread('test.jpeg')

# Pré-processa a imagem
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Executa a detecção de objetos
net.setInput(blob)
outputs = net.forward(net.getUnconnectedOutLayersNames())

# Processa os resultados
boxes = []
confidences = []
class_ids = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])
            h = int(detection[3] * img.shape[0])
            x = center_x - w // 2
            y = center_y - h // 2
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Desenha as caixas delimitadoras dos objetos
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Verifica se há objetos detectados
if len(indices) > 0:
    # Itera sobre os índices das caixas delimitadoras mantidas após a supressão não-máxima
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # Desenha a caixa delimitadora e a classe do objeto detectado
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, classes[class_ids[i]], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostra a imagem com os objetos detectados


# Salva a imagem com os objetos detectados
cv2.imwrite('output.jpg', img)
