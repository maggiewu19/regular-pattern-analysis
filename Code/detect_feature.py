from directory import *
from utility import *

def corner_detection(image, mask, unit, maxCorners=300, epsilon=1e-4, k=5e-2, block=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape 

    # harris corner detection 
    minDistance = 0.8*unit 
    corners = set()
    features = cv2.goodFeaturesToTrack(gray, maxCorners=maxCorners, 
                            qualityLevel=epsilon, minDistance=minDistance, 
                            useHarrisDetector=True, blockSize=block, k=k, mask=mask)
    minMaxPosition = (height, 0)
    for corner in np.int64(features): 
        x,y = corner.ravel()
        corners.add((int(x),int(y)))
        minMaxPosition = (min(y, minMaxPosition[0]), max(y, minMaxPosition[1]))
        cv2.circle(image, (x,y), 1, (255,0,0)) 
    
    return corners, minMaxPosition

def get_corner_info(corners, unit):
    # update feature vectors and neighbor info 
    featureVectors = dict()
    neighborInfo = dict()
    distance = 1.8*unit 
    minDifference = 0.25*unit 
    for x,y in corners: 
        featureVectors[(x,y)] = [0 for _ in range(8)]
        neighborInfo[(x,y)] = {'count': 0, 'coord': {index: None for index in range(8)}, 
                            'dist': [unit for _ in range(8)]}

        xRange = range(round(x-distance), round(x+distance))
        yRange = range(round(y-distance), round(y+distance))
        expectedNeighbors = [(x, y-unit), (x+unit, y-unit), (x+unit, y), 
                            (x+unit, y+unit), (x, y+unit), (x-unit, y+unit), 
                            (x-unit, y), (x-unit, y-unit)]

        for nx,ny in it.product(xRange, yRange):
            if (nx,ny) in corners and euclidean(nx, ny, x, y) > minDifference:
                neighborDistances = [euclidean(*en, nx, ny) for en in expectedNeighbors]
                neighborIndex = np.argmin(neighborDistances)
                
                if neighborDistances[neighborIndex] < neighborInfo[(x, y)]['dist'][neighborIndex]:
                    featureVectors[(x,y)][neighborIndex] = 1
                    neighborInfo[(x,y)]['coord'][neighborIndex] = (nx,ny)
                    neighborInfo[(x,y)]['dist'][neighborIndex] = neighborDistances[neighborIndex]
        
        neighborInfo[(x,y)]['count'] = sum(featureVectors[(x,y)])
    
    return featureVectors, neighborInfo 

def get_corner_matching(featureVectors, neighborInfo): 
    # find extended vector and corner matches 
    cornerMatching = dict()
    for x,y in neighborInfo: 
        neighbors = [neighborInfo[(x,y)]['coord'][index] for index in range(8)]
        extendedVector = "" 
        for neighbor in neighbors: 
            featureVector = "".join(str(sequence) for sequence in featureVectors.get(neighbor, '--------')) + ' '
            extendedVector += featureVector
        extendedVector = extendedVector[:-1]
    
        cornerMatching[(x,y)] = list()
        for corner in extensionInfo:
            cornerVector = extensionInfo[corner]

            length = len(extendedVector)
            missCount = int(sum([1 if extendedVector[i] != cornerVector[i] else 0 for i in range(length)]))
            cornerMissCount = int(sum([1 if ((extendedVector[i] == '-' or cornerVector[i] == '-') and extendedVector[i] != cornerVector[i]) else 0 for i in range(length)])/8)
            matchCount = None 
            valid = (cornerMissCount <= 2) and (missCount <= (cornerMissCount+1)*8)
            if valid: 
                cornerMatching[(x,y)].append((corner, missCount, cornerMissCount, matchCount))
        
        cornerMatching[(x,y)].sort(key=lambda x: x[1])

    return cornerMatching 

def section_split(minMaxPosition, cornerMatching):
    # section split 
    minPosition, maxPosition = minMaxPosition 
    section = (maxPosition-minPosition)/5 
    sectionSplits = [minPosition + section*n for n in range(6)]

    expectedSections = dict()
    startCorner = 0 
    for section in sectionInfo: 
        maxSectionCorner = sectionInfo[section]
        for corner in range(startCorner, maxSectionCorner+1):
            expectedSections[str(corner)] = int(section) 
        startCorner = maxSectionCorner 
    
    for x,y in cornerMatching: 
        sectionList = list()
        for corner, missCount, cornerMissCount, matchCount in cornerMatching[(x,y)]:
            expectedSection = expectedSections[corner]
            if y <= sectionSplits[0]: actualSection = 0 
            elif y >= sectionSplits[-1]: actualSection = len(sectionSplits)+1 
            else: 
                for sectionIndex in range(len(sectionSplits)):
                    if sectionSplits[sectionIndex] <= y < sectionSplits[sectionIndex+1]:
                        actualSection = sectionIndex+1 
                        break 
            
            if abs(actualSection-expectedSection) <= 1: 
                sectionList.append((corner, missCount, cornerMissCount, matchCount))
        
        cornerMatching[(x,y)] = sectionList 

    return cornerMatching 

def identify_matches(neighborInfo, cornerMatching):
    # identify corner matches 
    for x,y in cornerMatching: 
        identifyList = list()
        for corner, missCount, cornerMissCount, _ in cornerMatching[(x,y)]: 
            neighborCoords = neighborInfo[(x,y)]['coord']
            matchCount = 0 
            for index in range(8): 
                neighbor = neighborCoords[index] 
                if neighbor == None: continue 

                expectedNeighbor = cornerNeighborInfo[corner][index]
                for neighborCorner, _, _, _ in cornerMatching[neighbor]: 
                    if expectedNeighbor == neighborCorner: 
                        matchCount += 1 
                        break 
            identifyList.append((corner, missCount, cornerMissCount, matchCount))
        
        cornerMatching[(x,y)] = identifyList 
        cornerMatching[(x,y)].sort(key=lambda x: x[3], reverse=True)
    
    return cornerMatching 

def label_corners(image, identities, color, fontScale=0.5, cornerLabels=None, remove=False):
    if cornerLabels == None: cornerLabels = set()
    for x,y in identities: 
        if (x,y) not in cornerLabels: 
            corner = identities[(x,y)]
            cv2.putText(image, str(corner), org=(x-3*len(str(corner)), y-3), fontFace=cv2.FONT_HERSHEY_PLAIN, color=color, fontScale=fontScale)
            if not remove: cornerLabels.add((x,y))

    return cornerLabels

def assign_identities(image, cornerMatching):
    # assign corner identity 
    moreIdentities = False 
    identities = dict()
    takenCorners = dict()
    for x,y in cornerMatching: 
        if len(cornerMatching[(x,y)]) > 0: 
            corner, _, _, matchCount = cornerMatching[(x,y)][0] 
            if matchCount >= 2: 
                if corner in takenCorners: 
                    position, count = takenCorners[corner]
                    if count < matchCount: 
                        takenCorners[corner] = [(x,y), matchCount]
                        moreIdentities = True 
                        identities.pop(position)
                        identities[(x,y)] = corner 
                else: 
                    moreIdentities = True 
                    takenCorners[corner] = [(x,y), matchCount]
                    identities[(x,y)] = corner 

    return identities, takenCorners 

def extend_identities(image, neighborInfo, identities, takenCorners, minScore=2):
    # extend corner identity 
    moreIdentities = False 
    tempScores = dict()
    cornerScores = dict()
    for x,y in identities: 
        corner = identities[(x,y)]
        neighborCoords = neighborInfo[(x,y)]['coord'] 
        for index in range(8):
            neighbor = neighborCoords[index] 
            if neighbor == None: continue 
            if neighbor in identities: continue 
            
            tempScores[neighbor] = tempScores.get(neighbor, dict())
            expectedNeighbor = cornerNeighborInfo[corner][index]

            if expectedNeighbor != '-': 
                tempScores[neighbor][expectedNeighbor] = tempScores[neighbor].get(expectedNeighbor, 0) + 1

    for x,y in tempScores: 
        cornerScores[(x,y)] = list()
        for possibleCorner in tempScores[(x,y)]: 
            cornerScores[(x,y)].append((possibleCorner, tempScores[(x,y)][possibleCorner]))
        
        cornerScores[(x,y)].sort(key=lambda x: x[1], reverse=True)
    
    for x,y in cornerScores: 
        if len(cornerScores[(x,y)]) > 0: 
            corner, score = cornerScores[(x,y)][0] 
            if score >= minScore: 
                if corner in takenCorners: 
                    position, count = takenCorners[corner]
                    if count < score: 
                        takenCorners[corner] = [(x,y), score]
                        moreIdentities = True 
                        identities.pop(position)
                        identities[(x,y)] = corner 
                else: 
                    moreIdentities = True 
                    takenCorners[corner] = [(x,y), score]
                    identities[(x,y)] = corner 

    return identities, takenCorners, moreIdentities

def distance_check(unit, identities, takenCorners, count=0, maxCount=10):
    # check corner identity via distant neighbor 
    scores = dict()
    regionDistance = 1.5*unit
    farDistance = 2*unit 
    for x,y in identities: 
        corner = identities[(x,y)]

        top = set(distanceNeighborInfo[corner]["T"])
        down = set(distanceNeighborInfo[corner]["D"])
        left = set(distanceNeighborInfo[corner]["L"])
        right = set(distanceNeighborInfo[corner]["R"])

        for nx,ny in identities: 
            if (nx,ny) != (x,y): 
                neighborCorner = identities[(nx,ny)]
                if x-regionDistance < nx <= x+regionDistance and ny <= y-farDistance: 
                    if int(neighborCorner) in top: scores[(x,y)] = scores.get((x,y), 0) + 1 
                    else: scores[(x,y)] = scores.get((x,y), 0) - 1 
                if x-regionDistance < nx <= x+regionDistance and ny >= y+farDistance: 
                    if int(neighborCorner) in down: scores[(x,y)] = scores.get((x,y), 0) + 1 
                    else: scores[(x,y)] = scores.get((x,y), 0) - 1 
                if nx <= x-farDistance and y-regionDistance < ny <= y+regionDistance: 
                    if int(neighborCorner) in left: scores[(x,y)] = scores.get((x,y), 0) + 1 
                    else: scores[(x,y)] = scores.get((x,y), 0) - 1 
                if nx >= x+farDistance and y-regionDistance < ny <= y+regionDistance: 
                    if int(neighborCorner) in right: scores[(x,y)] = scores.get((x,y), 0) + 1 
                    else: scores[(x,y)] = scores.get((x,y), 0) - 1 

    if len(scores) == 0: 
        return 

    sortedScores = sorted(scores.items(), key=lambda x: x[1])
    if sortedScores[0][1] < 0 and count < maxCount: 
        corner = identities[sortedScores[0][0]]
        identities.pop(sortedScores[0][0])
        takenCorners.pop(corner)
        count += 1
        distance_check(unit, identities, takenCorners, count)

    return 

def iterative_extend(image, unit, neighborInfo, identities, takenCorners, cornerLabels, color=(0,150,0), minScore=2, maxCount=20):
    moreIdentities = True 
    count = 0
    while moreIdentities and count < maxCount: 
        identities, takenCorners, moreIdentities = extend_identities(image, neighborInfo, identities, takenCorners, minScore=minScore)
        distance_check(unit, identities, takenCorners) 
        cornerLabels = label_corners(image, identities, color, cornerLabels=cornerLabels)
        count += 1 
    
    return identities, takenCorners, cornerLabels
