import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")

#verificando se o video está aberto
if cap.isOpened():
    rval, frame = cap.read()
else:
    rval = False

while rval:

    img = frame

    #passando o vídeo para de bgr para hsv
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #criando as máscaras mínimas e máximas das cores amarelo e rosa
    img_low_hsv = np.array([160, 100, 100])
    img_up_hsv = np.array([180, 255, 255])
    img_low_hsv1 = np.array([20, 84, 100])
    img_up_hsv2 = np.array([40, 255, 255])

    #criando as máscaras
    mask1 = cv2.inRange(img_hsv, img_low_hsv, img_up_hsv)
    mask2 = cv2.inRange(img_hsv, img_low_hsv1, img_up_hsv2)

    #unindo as máscaras usando bitwise
    mask = cv2.bitwise_or(mask1, mask2)

    #encontrando os contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #ordenando os 2 maiores contornos do maior para o menos 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    #enquanto estiver em 2 contornos executa isso (2 maiores objetos)
    for i in range(2):
        #desenha o contorno dos objetos na imagem
        cv2.drawContours(img, [contours[i]], -1, (0, 0, 0), 2)
        #encontra os momentos para calcular as coordenadas x e y
        M = cv2.moments(contours[i])
        #se momentos for diferente de 0 executa o cálculo das coordenadas
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contours[i])

            #função para calcular o centro de nassa dos objetos
            def calculaCentro(contours):
                M = cv2.moments(contours)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return cx, cy
            cx1, cy1 = calculaCentro(contours[0])
            cx2, cy2 = calculaCentro(contours[1])

            #criando os deltas para calcular a inclinação da reta traçada
            delta_y = cy2 - cy1
            delta_x = cx2 - cx1
            #função do matemática do numpy 
            angle = np.degrees(np.arctan2(delta_y, delta_x))

            #traçando a reta
            cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            
            #criando cruz no centro dos objetos
            size = 10
            color = (0, 0, 0)
            cv2.line(img, (cx - size, cy), (cx + size, cy), color, 3)
            cv2.line(img, (cx, cy - size), (cx, cy + size), color, 3)
            
            #printando o objeto, sua área e seu centro de massa no console
            print("Objeto {}: Área={} Centro de massa=({}, {})".format(i+1, area, cx, cy))

            #formatando o texto para 2 casas decimais
            text = "Area: {:.2f}".format(area)

            #colocando o texto de área no vídeo
            cv2.putText(img, text, (cx-50, cy+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            #colocando o texto de centro de massa no vídeo
            text = "Centro: ({}, {})".format(cx, cy)
            cv2.putText(img, text, (cx-50, cy+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  

    #colcoando os graus de inclinação da reta no vídeo
    text = "Grau de inclinacao: {:.2f} ".format(angle)
    cv2.putText(frame, text, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    #mostrando o vídeo em uma janela
    cv2.imshow("Objetos segmentados", frame)
    rval, frame = cap.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow(frame)
cap.release()