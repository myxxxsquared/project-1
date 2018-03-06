import cv2
import postprocessing
import numpy as np
p = postprocessing.Postprocessor()
maps = np.load('maps.npy')
print(maps.shape)
ctns = p.process(maps)

cv2.imwrite('o.png', (maps[:, :, 0:1]*255).astype(np.uint8))

x = np.zeros(shape=(maps.shape[0], maps.shape[1]), dtype=np.uint8)
cv2.drawContours(x, ctns, -1, (255, 255, 255))

cv2.imwrite('o2.png', x)
