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
    
    # print ('Group Check: # identities = {}, # newIdentities = {}'.format(len(identities), len(newIdentities)))
    return newIdentities

def occlusion_check(identities, corners, unit):
    def check_range(cRange):
        return sum([1 if (nx,ny) in corners else 0 for nx,ny in cRange])

    newIdentities = dict()
    leftCorners = set(edgeCornerInfo['left'])
    rightCorners = set(edgeCornerInfo['right'])
    topCorners = set(edgeCornerInfo['top'])
    bottomCorners = set(edgeCornerInfo['bottom'])
    edgeCorners = leftCorners.union(rightCorners).union(topCorners).union(bottomCorners)

    for x,y in identities: 
        corner = identities[(x,y)]

        if corner in edgeCorners: 
            newIdentities[(x,y)] = corner
            continue 
        
        left = it.product(range(int(x-10*unit), int(x-0.25*unit)), range(int(y-0.5*unit), int(y+0.5*unit)))
        right = it.product(range(int(x+0.25*unit), int(x+10*unit)), range(int(y-0.5*unit), int(y+0.5*unit)))
        top = it.product(range(int(x-0.5*unit), int(x+0.5*unit)), range(int(y-10*unit), int(y-0.25*unit)))
        bottom = it.product(range(int(x-0.5*unit), int(x+0.5*unit)), range(int(y+0.25*unit), int(y+10*unit)))
        
        if min(check_range(right), check_range(left), check_range(top), check_range(bottom)) > 0: 
            newIdentities[(x,y)] = corner

    # print ('Occlusion Check: # identities = {}, # newIdentities = {}'.format(len(identities), len(newIdentities)))
    return newIdentities

def interpolate(rawImage, frame):
    prevData = load_pickle(ddst + '{}.pickle'.format(frame-1), dict())

    rect_matrix = prevData['rect-matrix']
    unit = prevData['unit']
    identities = prevData['identities']
    homography = prevData['homography']
    height, width, depth = rawImage.shape
    image = cv2.warpPerspective(rawImage, rect_matrix, (width, height), flags=cv2.INTER_CUBIC)

    return image, homography, rect_matrix, unit, identities

def image_transform(rawImage, image, frame, rect_matrix, unit, corners, identities, neighborInfo):
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
            identities = occlusion_check(identities, corners, unit)
            homography = get_homography(identities)
            meanDist, errorCount = image_error(identities, homography)
            errorStatus = f'Error Count = {errorCount}' if errorCount != 0 else ''
            print (f'Frame {frame}: Mean Dist = {meanDist} ' + errorStatus)
            if meanDist >= 10 or errorCount >= 20 or len(identities) <= 30: 
                print ('False Status', meanDist, errorCount, len(identities))
                raise ValueError
        except: 
            status = False
            image, homography, rect_matrix, unit, identities = interpolate(rawImage, frame)
    else: 
        status = False 
        image, homography, rect_matrix, unit, identities = interpolate(rawImage, frame)
    
    return image, rect_matrix, unit, identities, homography, status