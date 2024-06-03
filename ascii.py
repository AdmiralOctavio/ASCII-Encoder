import numpy as np
import pygame as pg
boxes = ['▫', '░', '▒', '▓', '█', '█']
boxes_grayscale = [0, 0.2, 0.4, 0.6, 0.8, 1]
boxes2 = ['.', '▫', '□', '▪', '■', '■'] #Terminal doesnt like these characters
characters = ['M', 'W', 'Q', 'B', 'E', 'R', 'N', '@', 'H', 'q', 'p', 'g', 'K', 'A', '#', 'm', 'b', '8', '0', 'd', 'X', 'D', 'G', 'F', 'P', 'e', 'h', 'U', '9', '6', 'k', 'Z', '%', 'S', '4', 'O', 'x', 'y', 'T', '5', 'w', 'f', 'a', 'V', 's','2', 'L', '$', 'Y', '&', 'n', '3', 'C', 'J', 'u', 'o', 'z', 'I', 'j', 'v', 'c', 'r', 't', 'l', 'i', '1', '=', '?', '7', '>', '<', ']', '[', '(', ')', '+', '*', ';', '}', '{', ':', '/', '\\', '!', '|', '_', ',', '^', '-', '~', '.', ' ']
grayscale = np.array([217.56944444, 218.82291667, 219.89236111, 220.19444444, 222.14583333, 222.94097222, 223.0625, 223.17361111, 223.22222222, 223.23958333, 223.45486111, 223.60416667,224.05208333, 224.09722222, 224.33333333, 225.25,225.59722222, 225.62152778, 225.91666667, 225.96180556,226.10763889, 226.74305556, 226.80208333, 227.04861111,227.42361111, 228.45833333, 228.61458333, 228.73958333,228.76736111, 228.80555556, 228.8125, 228.90625,228.98611111, 229.06597222, 229.28472222, 229.61805556,229.96527778, 230.07291667, 230.17361111, 230.21875,230.60416667, 230.62847222, 230.84375, 231.03472222,231.05555556, 231.46875, 231.55555556, 231.9375,232.04861111, 232.07291667, 232.64583333, 232.68055556,233.16319444, 233.53472222, 233.70138889, 234.20833333,234.40625, 234.76388889, 234.93055556, 235.30208333,235.36805556, 235.44791667, 235.5, 236.53472222,237.32986111, 237.67361111, 237.70138889, 238.61458333,238.61805556, 238.78125, 238.78472222, 238.79166667,238.98611111, 239.07638889, 239.08680556, 239.97569444,240.32291667, 240.78125, 241.50694444, 241.57291667,242.25694444, 243.13194444, 243.18055556, 243.31944444,244.30208333, 244.61805556, 245.03819444, 246.62847222,247.58333333, 247.60763889, 248.62847222, 255.0])
grayscale = np.clip((grayscale - np.min(grayscale))/(np.max(grayscale) - np.min(grayscale)), 0, 1) #Normalised grayscale values; clip any anomalous values
char_width = 2*1
char_height = 1*1
char_hw_ratio = char_height / char_width
image = pg.image.load("image16.jpg")
grayImg = pg.surfarray.pixels_red(image)*0.33 + pg.surfarray.pixels_blue(image)*0.33 + pg.surfarray.pixels_green(image)*0.34

def nearest(array, value): 
    '''Returns index of the nearest element in the array to the given value'''
    array = np.asarray(array)
    index = (np.abs(array-value)).argmin()
    return index

inp = input("EnTER MODE")
if inp == '1':
    findNormalisation = []
    for i in range(0, grayImg.shape[1]-char_width, char_width):
        for j in range(0, grayImg.shape[0]-char_height, char_height):
            kern = grayImg[j:j+char_width, i:i+char_height]
            findNormalisation.append(np.sum(kern)/(char_height*char_width))

    minimum = min(findNormalisation)
    maximum = max(findNormalisation)

    for i in range(0, grayImg.shape[1]-char_width, char_width):
        bigTing = []
        for j in range(0, grayImg.shape[0]-char_height, char_height):
            kernel = grayImg[j:j+char_width, i:i+char_height]
            kernel = np.clip((kernel - minimum)/(maximum - minimum), 0, 1)
            kernel = np.sum(kernel)/(char_height*char_width)
            nearestVal = nearest(grayscale, kernel)
            bigTing.append(characters[len(characters)-1-nearestVal])
        print(''.join(bigTing))
else:
    for i in range(0, grayImg.shape[1]-char_width, char_width):
        bigTing = []
        for j in range(0, grayImg.shape[0]-char_height, char_height):
            kernel = grayImg[j:j+char_width, i:i+char_height]
            kernel = np.sum(kernel)/(char_height*char_width) /255
            nearestVal = nearest(boxes_grayscale, kernel) 
            bigTing.append(boxes[nearestVal])
        print(''.join(bigTing))
