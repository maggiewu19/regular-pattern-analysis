from utility import *
from detect_feature import *
from find_homography import * 

root = '/Users/maggiewu/Documents/Post_MEng_Research/'
info_dir = root + 'Maze_Info/'
pixelInfo = load_json(info_dir + 'corner_pixels_info.json')

def temporal_analysis(image, unit, corners, identities, cornerLabels, frame_data):
    if len(frame_data['identities']) == 0: return identities, cornerLabels 

    homography = get_homography(frame_data['identities'], identities2=identities)
    identityCorners = set(identities.values())
    distance = 0.5*unit 

    for x,y in frame_data['identities']: 
        corner = frame_data['identities'][(x,y)]
        if corner not in identityCorners: 
            frameX, frameY = transform_coord(x, y, homography)
            xRange = range(round(frameX-distance), round(frameX+distance))
            yRange = range(round(frameY-distance), round(frameY+distance))
            
            dist = distance 
            matchX, matchY = None, None 
            for nx,ny in it.product(xRange, yRange):
                if (nx,ny) in corners and (nx,ny) not in identities and euclidean(frameX, frameY, nx, ny) < dist: 
                    dist = euclidean(frameX, frameY, nx, ny)
                    matchX, matchY = nx, ny 
            
            if matchX != None and matchY != None: 
                identities[(matchX, matchY)] = corner 

    cornerLabels = label_corners(image, identities, (150,0,150), cornerLabels=cornerLabels)

    return identities, cornerLabels 

def template_match(image, unit, corners, identities, cornerLabels, homography):
    identityCorners = set(identities.values())
    distance = 0.5*unit

    for corner in pixelInfo: 
        if corner not in identityCorners: 
            x,y = pixelInfo[corner] 
            imageX, imageY = transform_coord(x, y, np.linalg.inv(homography))

            xRange = range(round(imageX-distance), round(imageX+distance))
            yRange = range(round(imageY-distance), round(imageY+distance))
            
            dist = distance 
            matchX, matchY = None, None 
            for nx,ny in it.product(xRange, yRange):
                if (nx,ny) in corners and (nx,ny) not in identities and euclidean(imageX, imageY, nx, ny) < dist: 
                    dist = euclidean(x, y, nx, ny) 
                    matchX, matchY = nx, ny 

            if matchX != None and matchY != None: 
                identities[(matchX, matchY)] = corner 

    cornerLabels = label_corners(image, identities, (150,150,0), cornerLabels=cornerLabels)

    return identities, cornerLabels 