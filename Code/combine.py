from directory import *
from conditions import *
from deblur_image import *
from rectify_image import *
from create_mask import *
from detect_feature import * 
from find_homography import * 
from post_processing import *
from icp import * 

import time 

def preprocess(image, frame, prevData, interval=5):
    '''
    Image rectification via hough transform 
    Unit estimation based on hough lines 

    Input: image (np.array)
            low (int) 
            high (int)
    Output: image (np.array)
            mask (np.array)
            unit (float)
    '''

    def run(height, width, depth, homography, unit):
        try: 
            lines, n = find_lines(image.astype(np.int16))
            nhx, nhy, nvx, nvy, hdismiss, vdismiss = find_vanishing(image, lines, n)
            nunit = find_unit(lines, n, hdismiss, vdismiss)
            nhomography = compute_rectify_matrix(height, width, depth, nhx, nhy, nvx, nvy)
            exp_conditions['prevInter'] = False
            return nhomography, nunit 
        except: 
            exp_conditions['prevInter'] = True
            return homography, unit 

    height, width, depth = image.shape
    homography = prevData.get('rect-matrix', np.identity(3))
    unit = prevData.get('unit', 0)

    if exp_conditions['prevInter'] or not frame % interval: 
        homography, unit = run(height, width, depth, homography, unit)

    image = cv2.warpPerspective(image, homography, (width, height), flags=cv2.INTER_CUBIC)

    return image, homography, unit 

def bias_icp(corners, unit, ground, prevCorners, prevHomography):
    biasCorners = []
    for x,y in corners:
        xRange = range(round(x-2*unit), round(x+2*unit))
        yRange = range(round(y-2*unit), round(y+2*unit))
        for nx,ny in it.product(xRange, yRange):
            if (nx,ny) in prevCorners: 
                biasCorners.append([x,y])
                break 

    homography = icp(np.array(biasCorners), ground, init_pose=prevHomography)
    
    return homography 

def compute_icp(corners, ground, prevHomography):
    corners = np.array([[x,y] for x,y in corners])
    homography = icp(corners, ground, init_pose=prevHomography)
    return homography 

def analyze_neighbor(featureVectors, neighborInfo, minMaxPosition):
    cornerMatching = get_corner_matching(featureVectors, neighborInfo)
    cornerMatching = section_split(minMaxPosition, cornerMatching)

    return cornerMatching

def analyze_region(image, unit, cornerMatching, neighborInfo):
    '''
    Corner detection via Harris corner detector 
    Corner matching and regional pattern analysis 

    Input: image (np.array)
            unit (float)
    Output: image (np.array)
            identities (dict) 
    '''
    cornerMatching = identify_matches(neighborInfo, cornerMatching)
    identities, takenCorners = assign_identities(cornerMatching) 
    distance_check(unit, identities, takenCorners) 
    red = copy.deepcopy(identities)
    
    identities, takenCorners = iterative_extend(image, unit, neighborInfo, identities, takenCorners)
    green = copy.deepcopy(identities)
    
    identities, takenCorners = iterative_extend(image, unit, neighborInfo, identities, takenCorners, minScore=1)
    blue = copy.deepcopy(identities)

    return image, identities, red, green, blue

def transform_update(rawImage, image, frame, rect_matrix, unit, corners, identities, neighborInfo):
    '''
    Perspective transformation via homography mapping 

    Input: image (np.array)
            identities (dict) 
    Output: image (np.array)
    '''
    return image_transform(rawImage, image, frame, rect_matrix, unit, corners, identities, neighborInfo)

def sequential_label(image, red, green, blue, purple, everything, fontScale=0.5):
    def label(target, color, labelled):
        for x,y in target: 
            if (x,y) in everything and (x,y) not in labelled and target[(x,y)] == everything[(x,y)]: 
                labelled.add((x,y))
                corner = everything[(x,y)]
                cv2.putText(image, str(corner), org=(x-3*len(str(corner)), y-3), fontFace=cv2.FONT_HERSHEY_PLAIN, color=color, fontScale=fontScale) 
        return labelled 

    labelled = label(red, (0,0,255), set())
    labelled = label(green, (0,150,0), labelled)
    labelled = label(blue, (255,0,0), labelled)
    labelled = label(purple, (150,0,150), labelled)
    labelled = label(everything, (150,150,0), labelled)

    return image 
    
def log(frame, image, homographyImage, rect_matrix, unit, identities, homography, save, logging, show):
    frameData = dict()
    frameData['rect-matrix'] = rect_matrix
    frameData['unit'] = unit 
    frameData['identities'] = identities 
    frameData['homography'] = homography 
    
    if show: 
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    if save: cv2.imwrite(cdst + '{}.jpg'.format(frame), image)
    if save: cv2.imwrite(hdst + '{}.jpg'.format(frame), homographyImage)
    if logging: save_pickle(ddst + '{}.pickle'.format(frame), frameData)

def pipeline(frame, image, mask, ground, show, save, logging, imageUnit=48):
    # load image and previous data 
    rawImage = copy.deepcopy(image) 
    prevData = load_pickle(ddst + '{}.pickle'.format(frame-1), dict())

    # image rectification 
    image, rect_matrix, unit = preprocess(image, frame, prevData)

    # corner detection 
    corners, minMaxPosition = corner_detection(image, mask, unit)
    featureVectors, neighborInfo = get_corner_info(corners, unit)

    # run icp algorithms
    icpCornerMatching = dict()
    if prevData: 
        prevHomography = prevData.get('homography', None)
        prevCorners = set(prevData.get('identities', dict()).keys())
        if prevCorners: 
            icpHomography = bias_icp(corners, unit, ground, prevCorners, prevHomography)
        else: 
            icpHomography = compute_icp(corners, ground, prevHomography)
        
        pixelCorner = dict()
        for pixel in pixelCornerInfo: 
            x,y = pixel.split(',')
            pixelCorner[(int(x), int(y))] = pixelCornerInfo[pixel]

        icpIdentities = dict()
        distance = 0.75*imageUnit
        for x,y in corners: 
            icpList = list()
            ix, iy = transform_coord(x, y, icpHomography)
            xRange = range(round(ix-distance), round(ix+distance))
            yRange = range(round(iy-distance), round(iy+distance))
            for nx,ny in it.product(xRange, yRange):
                if (nx,ny) in pixelCorner: 
                    icpList.append((pixelCorner[(nx,ny)], euclidean(x,y,nx,ny)))
            if len(icpList) > 0: 
                icpIdentities[(x,y)] = sorted(icpList, key=lambda x: x[1], reverse=True)[0][0]

        icpCornerMatching = analyze_icp(icpIdentities, featureVectors, neighborInfo)

    neighborCornerMatching = analyze_neighbor(featureVectors, neighborInfo, minMaxPosition)

    if icpCornerMatching:
        cornerMatching = dict()
        matches = set(icpCornerMatching.keys()).union(set(neighborCornerMatching.keys()))
        for x,y in matches: 
            if (x,y) in icpCornerMatching and (x,y) in neighborCornerMatching: 
                icorner, imissCount, icornerMissCount, imatchCount = icpCornerMatching[(x,y)][0]
                findMatch = False 
                for ncorner, nmissCount, ncornerMissCount, nmatchCount in neighborCornerMatching[(x,y)]: 
                    if icorner == ncorner: 
                        if (imissCount-icornerMissCount*8) < (nmissCount-ncornerMissCount*8): 
                            cornerMatching[(x,y)] = [(icorner, imissCount, icornerMissCount, imatchCount)]
                        else: 
                            cornerMatching[(x,y)] = [(ncorner, nmissCount, ncornerMissCount, nmatchCount)]
                        findMatch = True 
                        break 
                if not findMatch: 
                    bncorner, bnmissCount, bncornerMissCount, bnmatchCount = neighborCornerMatching[(x,y)][0]
                    if (imissCount-icornerMissCount*8) < (bnmissCount-bncornerMissCount*8): 
                        cornerMatching[(x,y)] = [(icorner, imissCount, icornerMissCount, imatchCount)]
                    else: 
                        cornerMatching[(x,y)] = [(bncorner, bnmissCount, bncornerMissCount, bnmatchCount)]
            elif (x,y) in icpCornerMatching: 
                cornerMatching[(x,y)] = icpCornerMatching[(x,y)]
            elif (x,y) in neighborCornerMatching:
                cornerMatching[(x,y)] = neighborCornerMatching[(x,y)]
            else: 
                pass 
    else:
        cornerMatching = neighborCornerMatching

    # print (cornerMatching)
    
    image, identities, red, green, blue = analyze_region(image, unit, cornerMatching, neighborInfo)
    identities = temporal_analysis(image, frame, unit, corners, identities)
    purple = copy.deepcopy(identities)

    image, rect_matrix, unit, identities, homography, status = transform_update(rawImage, image, frame, rect_matrix, unit, corners, identities, neighborInfo) 
    everything = copy.deepcopy(identities)
    if status: image = sequential_label(image, red, green, blue, purple, everything)
    homographyImage = warp_image(image, homography)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # log(frame, image, homographyImage, rect_matrix, unit, identities, homography, save=save, logging=logging, show=show)

def main():
    frameRange = range(358, 359)

    for frame in frameRange: 
        fname = src + 'frame{}.png'.format(frame)
        image = cv2.imread(fname)
        ground = np.array(list(pixelInfo.values()))

        if frame == frameRange[0]: 
            brightLow, brightHigh = select_region(image)
            brightMask = get_mask(image, brightLow, brightHigh)

            shadowLow, shadowHigh = select_region(image)
            shadowMask = get_mask(image, shadowLow, shadowHigh)

        print ('mask number')
        print (brightLow, brightHigh, shadowLow, shadowHigh)

        mask = cv2.bitwise_or(brightMask, shadowMask)

        cv2.imshow('mask', cv2.bitwise_and(image, image, mask=mask))

        try: 
            pipeline(frame, image, mask, ground, show=exp_conditions['show'] , save=exp_conditions['save'] , logging=exp_conditions['logging'])
        except: 
            print ('except pipeline')
            image, homography, rect_matrix, unit, identities = interpolate(image, frame)
            homographyImage = warp_image(image, homography)
            # log(frame, image, homographyImage, rect_matrix, unit, identities, homography, save=exp_conditions['save'], logging=exp_conditions['logging'], show=exp_conditions['show'])
            exp_conditions['prevInter'] = True 

main()