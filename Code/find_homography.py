from utility import *

root = '/Users/maggiewu/Documents/Post_MEng_Research/'
info_dir = root + 'Maze_Info/'
pixelInfo = load_json(info_dir + 'corner_pixels_info.json')

def get_homography(identities, identities2=None):
    if identities2 == None: 
        actualPoints = np.array(list(identities.keys()))
        expectedPoints = np.array([pixelInfo[identities[tuple(coord)]] for coord in actualPoints])
    else: 
        commonCorners = set(identities.values()).intersection(set(identities2.values()))
        actualPoints = [] 
        expectedPoints = []

        for corner in commonCorners: 
            for x1,y1 in identities: 
                if identities[(x1,y1)] == corner: actualPoints.append((x1,y1))
            for x2,y2 in identities2: 
                if identities2[(x2,y2)] == corner: expectedPoints.append((x2,y2)) 

        assert len(actualPoints) == len(expectedPoints)

        actualPoints = np.array(actualPoints)
        expectedPoints = np.array(expectedPoints)

    try: 
        homography, _ = cv2.findHomography(actualPoints, expectedPoints, method=cv2.LMEDS)
        return homography
    except:
        return None 

def image_error(identities, homography):
    transformedPoints = np.array([transform_coord(x, y, homography) for x,y in identities])
    expectedPoints = np.array([pixelInfo[identities[coord]] for coord in identities])

    meanDist, errorCount = distance_error(transformedPoints, expectedPoints)

    return meanDist, errorCount 

def warp_image(image, homography): 
    return cv2.warpPerspective(image, homography, (825, 1275))
