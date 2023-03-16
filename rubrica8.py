import cv2
import numpy as np

img = cv2.imread('circulo.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#criando as mascaras para vermelho e azul
image_lower_hsv1 = np.array([60,165,225])
image_upper_hsv1 = np.array([87,166,227])
image_lower_hsv2 = np.array([0,130,100])
image_upper_hsv2 = np.array([30,255,255])

#unindo as duas mascaras de minimo e máximo para as cores
mask1 = cv2.inRange(img_hsv, image_lower_hsv1, image_upper_hsv1)
mask2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)

#fazendo a junção das mascaras das duas cores
mask = cv2.bitwise_or(mask1, mask2)

#identificando o contorno dos objetos
contornos, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for i in range(2):
    cv2.drawContours(img, [contornos[i]], -1, (0, 255, 0), 2)

    M = cv2.moments(contornos[i])
    area = M['m00']
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    def calcula_centro_massa(contours):
        M = cv2.moments(contours)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    cx1, cy1 = calcula_centro_massa(contornos[0])
    cx2, cy2 = calcula_centro_massa(contornos[1])

    delta_y = cy2 - cy1
    delta_x = cx2 - cx1
    angulo = np.degrees(np.arctan2(delta_y, delta_x))

    size = 10
    color = (0, 0, 0)
    
    cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, 3)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, 3)
    text = "Inclinacao: {:.2f} graus".format(angulo)
    cv2.putText(img, text, (cx1-600, cy-370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    print("Objeto segmentado [0]: Área={} Centro de massa=({}, {})".format(i+1, area, cx, cy))
    text = "Area: {:.2f}".format(area)
    cv2.putText(img, text, (cx-50, cy+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    text = "Centro: ({}, {})".format(cx, cy)
    cv2.putText(img, text, (cx-50, cy+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv2.imshow("Objetos Segmentados", img)
cv2.waitKey(0)
cv2.destroyAllWindows()