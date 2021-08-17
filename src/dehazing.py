import image_dehazer
import cv2

HazeOriginal = cv2.imread('../inputs/pollutionHaze.jpg')
HazeOriginal = image_dehazer.remove_haze(HazeOriginal) 
cv2.imwrite('../inputs/pollutionDehaze.jpg',HazeOriginal)