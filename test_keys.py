import cv2
import numpy as np

img = np.zeros((200, 400, 3), np.uint8)
cv2.putText(img, "Nacisnij polskie znaki (Q = wyjscie)",
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

cv2.namedWindow("Test klawiszy")
cv2.imshow("Test klawiszy", img)

while True:
    raw_key = cv2.waitKeyEx(0)
    if raw_key == -1:
        continue
    key = raw_key & 0xFF
    print(f"raw_key={raw_key}  |  key&0xFF={key}  |  char={repr(chr(key) if key < 256 else '?')}")
    if key == ord('q'):
        break

cv2.destroyAllWindows()