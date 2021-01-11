from directory import *
from utility import *
from rectify_image import *
from detect_feature import *
from find_homography import * 

def sanity_check(identities):
    return len(identities) > (1/4) * len(pixelInfo)

def temporal_analysis(image, unit, corners, identities, cornerLabels, prevIdentities):
    if len(prevIdentities) == 0: return identities, cornerLabels 

    homography = get_homography(prevIdentities, identities2=identities)
    if not isinstance(homography, np.ndarray): return identities, cornerLabels 
    identityCorners = set(identities.values())
    distance = 0.5*unit 

    for x,y in prevIdentities: 
        corner = prevIdentities[(x,y)]
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

def remove_identities(image, unit, identities, homography):
    distance = 2*unit
    newIdentities = dict()
    newCornerLabels = set()

    for x,y in identities: 
        corner = identities[(x,y)]
        groundX, groundY = pixelInfo[corner] 
        imageX, imageY = transform_coord(groundX, groundY, np.linalg.inv(homography))

        if euclidean(imageX, imageY, x, y) <= distance: 
            newIdentities[(x,y)] = corner 
            newCornerLabels.add((x,y))
    
    label_corners(image, identities, (0,0,0), cornerLabels=newCornerLabels, remove=True)

    return newIdentities, newCornerLabels

def interpolate(rawImage, frame, frameData):
    print ('frame {} used previous information'.format(frame))
    hx, hy, vx, vy = frameData['vanishing']
    vanishing = [hx, hy, vx, vy]
    unit = frameData['unit']
    identities = dict() 
    homography = frameData['homography']
    image = rectify(rawImage, hx, hy, vx, vy)

    return image, homography, vanishing, unit, identities

def image_transform(rawImage, image, frame, vanishing, unit, corners, identities, cornerLabels, frameData, interpolationData):
    status = True 
    if sanity_check(identities): 
        try: 
            identities, cornerLabels = remove_identities(image, unit, identities, get_homography(identities))
            identities, cornerLabels = template_match(image, unit, corners, identities, cornerLabels, get_homography(identities))
            identities, cornerLabels = remove_identities(image, unit, identities, get_homography(identities))
            homography = get_homography(identities)
            meanDist, errorCount = image_error(identities, homography)
            print ('Mean Dist: {}, Error Count: {}'.format(meanDist, errorCount))
            if meanDist >= 10 or errorCount >= 20: raise ValueError
        except: 
            status = False
            image, homography, vanishing, unit, identities = interpolate(rawImage, frame, frameData)
    else: 
        status = False 
        image, homography, vanishing, unit, identities = interpolate(rawImage, frame, frameData)

    homographyImage = warp_image(image, homography)
    
    return homographyImage, image, vanishing, unit, identities, homography, status