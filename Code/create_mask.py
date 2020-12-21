from utility import *

def select_region(image):
    '''
    Select forground color to keep based on points
    Use to create mask for the image 

    Input: image (np.array)
    Output: low (np.array) 
            high (np.array)
    '''
    def onclick(click):
        point = int(round(click.xdata)), int(round(click.ydata))
        points.append(point)

    points = list()
    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = np.array([255, 255, 255])
    high = np.array([0, 0, 0])

    for x, y in points: 
        high = np.maximum(hsv[y][x], high)
        low = np.minimum(hsv[y][x], low)

    low = np.maximum(np.array([0, 0, 0]), low-10)
    high = np.minimum(np.array([255, 255, 255]), high+10)

    return low, high 

def get_mask(image, low, high):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.dilate(cv2.inRange(hsv, low, high), np.ones((15, 15), np.uint8))

    return mask 