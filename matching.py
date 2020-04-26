import cv2
import numpy as np
import math

imageArr = []
imageGrey = []
exhaustives = []
logarithm = []
hierar = []


def readFrame(filename):
    vidcap = cv2.VideoCapture(filename)
    success = True
    count = 0

    while success:
        success, image = vidcap.read()
        if success:
            imageArr.append(image)
            imageGrey.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        count += 1


def convertPixel():
    for i in range(len(imageGrey)):
        imageGrey[i] = 255 - imageGrey[i]


def initialframe():
    test = imageGrey[0].copy()
    indexX, indexY = -1, -1
    init = -math.inf

    for i in range(test.shape[0] - reference.shape[0]):
        for j in range(test.shape[1] - reference.shape[1]):
            temp = test[i:i + referenceRow, j:j + referenceColumn]
            ans = np.sum(temp * reference)

            if (ans > init):
                init = ans
                indexX, indexY = i, j

    print (indexX)
    print(indexY)
    return indexX, indexY


def eachFrame(frame, ref, p, x, y):
    sR = x - p
    eR = x + p
    sC = y - p
    eC = y + p

    rr, rc = ref.shape
    fr, fc = frame.shape

    indexX, indexY = -1, -1
    init = -math.inf

    if (sR < 0):
        sR = 0

    if eR + rr > fr:
        eR = fr - rr

    if sC < 0:
        sC = 0

    if eC + rc > fc:
        eC = fc - rc

    for i in range(sR, eR):
        for j in range(sC, eC):
            temp = frame[i:i + rr, j:j + rc]
            ans = np.sum(temp * ref)

            if (ans > init):
                init = ans
                indexX, indexY = i, j

    return indexX, indexY


def exhaustive(p):
    x, y = initialframe()
    coloredFrame = imageArr[0].copy()
    cv2.rectangle(coloredFrame, (y, x), (y + referenceColumn, x + referenceRow), (0, 0, 255), 5)
    exhaustives.append(coloredFrame)

    for i in range(1, len(imageGrey)):
        frame = imageGrey[i].copy()
        coloredFrame = imageArr[i].copy()
        x, y = eachFrame(frame, reference, p, x, y)

        cv2.rectangle(coloredFrame, (y, x), (y + referenceColumn, x + referenceRow), (0, 0, 255), 5)
        exhaustives.append(coloredFrame)


def video():
    height, width, layers = imageArr[0].shape
    video = cv2.VideoWriter('output.mov', cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

    for img in exhaustives:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def eachFrameLog(frame, p, x, y):
    k = math.ceil(math.log(p, 2))
    d = 2 ** (k - 1)
    init = - math.inf
    indexX, indexY = -1, -1

    for i in range(k):
        print("d= {}".format(d))

        coord = []

        for j in range(-1, 2):
            for k in range(-1, 2):
                row, column = x + j * d, y + k * d
                coord.append((row, column))

        for r, c in coord:

            r, c = int(r), int(c)

            if (r < 0):
                r = 0

            if r + referenceRow > frameRow:
                r = int(frameRow - referenceRow)

            if c < 0:
                c = 0

            if c + referenceColumn > frameCol:
                c = int(frameCol - referenceColumn)

            temp = frame[r:r + referenceRow, c:c + referenceColumn]
            ans = np.sum(temp * reference)

            if (ans > init):
                init = ans
                indexX, indexY = r, c

        d = d / 2

    return indexX, indexY


def logarithm2D(p):
    x, y = initialframe()
    coloredFrame = imageArr[0].copy()
    cv2.rectangle(coloredFrame, (y, x), (y + referenceColumn, x + referenceRow), (0, 0, 255), 5)
    logarithm.append(coloredFrame)

    for i in range(1, len(imageGrey)):
        frame = imageGrey[i]
        coloredFrame = imageArr[i].copy()

        x, y = eachFrameLog(frame, p, x, y)

        print((x, y))

        cv2.rectangle(coloredFrame, (y, x), (y + referenceColumn, x + referenceRow), (0, 0, 255), 5)
        logarithm.append(coloredFrame)


def videoLog():
    height, width, layers = imageArr[0].shape
    video = cv2.VideoWriter('log.mov', cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

    for img in logarithm:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def eachFramehierarchy(frame, p, x, y):
    l1 = cv2.pyrDown(frame)
    l2 = cv2.pyrDown(l1)
    r1 = cv2.pyrDown(reference)
    r2 = cv2.pyrDown(r1)
    v = int(p / 4)

    initx, inity = int(x / 4), int(y / 4)

    nx, ny = eachFrame(l2, r2, v, initx, inity)

    nx, ny = eachFrame(l1, r1, 1, nx * 2, ny * 2)

    fx, fy = eachFrame(frame, reference, 1, nx * 2, ny * 2)

    return fx, fy


def hierarchy(p):
    x, y = initialframe()
    coloredFrame = imageArr[0].copy()
    cv2.rectangle(coloredFrame, (y, x), (y + referenceColumn, x + referenceRow), (0, 0, 255), 5)
    hierar.append(coloredFrame)

    for i in range(1, len(imageGrey)):
        frame = imageGrey[i]
        coloredFrame = imageArr[i].copy()

        x, y = eachFramehierarchy(frame, p, x, y)

        print((x, y))

        cv2.rectangle(coloredFrame, (y, x), (y + referenceColumn, x + referenceRow), (0, 0, 255), 5)
        hierar.append(coloredFrame)


def videohierarchy():
    height, width, layers = imageArr[0].shape
    video = cv2.VideoWriter('hierarchy.mov', cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

    for img in hierar:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def result(fr, fc):
    ce = fr*fc
    cl = fr*fc
    ch = fr*fc

    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for i in p:
        ce += (i*i)*(len(imageGrey)-1)
        cl += (math.ceil(math.log(i,2))*8 + 1) * (len(imageGrey)-1)
        ch += (math.ceil(math.log(int(i/4),2))*8 + 8*2)*(len(imageGrey)-1)
        print("For p = {} : exhaustive == {}, logarithm = {} , hierarchical = {}".format(i,ce/(len(imageGrey)),cl/(len(imageGrey)),ch/(len(imageGrey))))


readFrame('movie.mov')
reference = cv2.imread('reference.jpg', 0)
referenceRow, referenceColumn = reference.shape
frameRow, frameCol = imageGrey[0].shape
convertPixel()

p = 20

exhaustive(p)
video()

logarithm2D(p)
videoLog()

hierarchy(p)
videohierarchy()

result(frameRow, frameCol)