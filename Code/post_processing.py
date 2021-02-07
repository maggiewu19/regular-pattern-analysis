from directory import *
from utility import *
from rectify_image import *
from detect_feature import *
from find_homography import * 

def sanity_check(identities):
    return len(identities) > (1/4) * len(pixelInfo)

def temporal_analysis(image, frame, unit, corners, identities):
    prevData = load_pickle(ddst + '{}.pickle'.format(frame-1), dict())
    prevIdentities = prevData.get('identities', dict())

    if len(prevIdentities) == 0: return identities 

    homography = get_homography(prevIdentities, identities2=identities)
    if not isinstance(homography, np.ndarray): return identities 
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

    return identities 

def template_match(image, unit, corners, identities, homography):
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

    return identities 

def remove_identities(image, unit, identities, homography):
    distance = 2*unit
    newIdentities = dict()

    for x,y in identities: 
        corner = identities[(x,y)]
        groundX, groundY = pixelInfo[corner] 
        imageX, imageY = transform_coord(groundX, groundY, np.linalg.inv(homography))

        if euclidean(imageX, imageY, x, y) <= distance: 
            newIdentities[(x,y)] = corner 

    return newIdentities

def alignment_check(image, unit, identities):
    def check(array, index, identities):
        for i in array: 
            position = [entry[index] for entry in array[i]]
            median = statistics.median(position) 
            for entry in array[i]: 
                if abs(entry[index] - median) > distance and entry in identities: 
                    del identities[entry]
        
        return identities 

    distance = 0.5*unit 
    groundHorizontal = alignmentInfo['horizontal']
    groundVertical = alignmentInfo['vertical']
    horizontal = dict()
    vertical = dict()

    for x,y in identities: 
        corner = identities[(x,y)] 
        for row in groundHorizontal: 
            if int(corner) in groundHorizontal[row]: 
                horizontal[row] = horizontal.get(row, []) + [(x,y)]
        for col in groundVertical: 
            if int(corner) in groundVertical[col]: 
                vertical[col] = vertical.get(col, []) + [(x,y)]

    identities = check(vertical, 0, identities)
    identities = check(horizontal, 1, identities)

    return identities

def group_check(identities, neighborInfo, minScore=2):
    newIdentities = dict()
    for x,y in identities: 
        corner = identities[(x,y)]
        actualNeighborsCount = sum([1 if c != '-' else 0 for c in cornerNeighborInfo[corner]])
        correctNeighborsCount = sum([1 if c in identities and identities[c] in cornerNeighborInfo[corner] else 0 for _, c in neighborInfo[(x,y)]['coord'].items()])
        if correctNeighborsCount >= min(minScore, 0.5*actualNeighborsCount): 
            newIdentities[(x,y)] = corner 
    
    print ('Group Check: # identities = {}, # newIdentities = {}'.format(len(identities), len(newIdentities)))

    return newIdentities

def interpolate(rawImage, frame):
    print ('frame {} used previous information'.format(frame))
    prevData = load_pickle(ddst + '{}.pickle'.format(frame-1), dict())

    hx, hy, vx, vy = prevData['vanishing']
    vanishing = [hx, hy, vx, vy]
    unit = prevData['unit']
    identities = dict() 
    homography = prevData['homography']
    image = rectify(rawImage, hx, hy, vx, vy)

    return image, homography, vanishing, unit, identities

def image_transform(rawImage, image, frame, vanishing, unit, corners, identities, neighborInfo, interpolationData):
    def filter_identity(image, unit, identities, neighborInfo):
        identities = remove_identities(image, unit, identities, get_homography(identities))
        identities = alignment_check(image, unit, identities)
        identities = group_check(identities, neighborInfo)
        return identities 
    
    status = True 
    if sanity_check(identities): 
        try: 
            identities = filter_identity(image, unit, identities, neighborInfo)
            identities = template_match(image, unit, corners, identities, get_homography(identities))
            identities = filter_identity(image, unit, identities, neighborInfo)
            homography = get_homography(identities)
            meanDist, errorCount = image_error(identities, homography)
            print ('Mean Dist: {}, Error Count: {}'.format(meanDist, errorCount))
            if meanDist >= 10 or errorCount >= 20 or len(identities) <= 30: raise ValueError
        except: 
            status = False
            image, homography, vanishing, unit, identities = interpolate(rawImage, frame)
    else: 
        status = False 
        image, homography, vanishing, unit, identities = interpolate(rawImage, frame)
    
    return image, vanishing, unit, identities, homography, status