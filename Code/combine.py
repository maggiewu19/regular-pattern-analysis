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

def preprocess(image, frame, low, high, prevData, prevInter=False, fast=False, interval=5):
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

            return nhomography, nunit, False 
        except: 

            return homography, unit, True 

    height, width, depth = image.shape
    homography = prevData.get('rect-matrix', np.identity(3))
    unit = prevData.get('unit', 0)
    prevInter = True 

    if prevInter or not frame % interval: 
        homography, unit, prevInter = run(height, width, depth, homography)

    image = cv2.warpPerspective(image, homography, (width, height), flags=cv2.INTER_CUBIC)
    mask = get_mask(image, low, high)

    return image, mask, homography, unit, prevInter 

def bias_icp(corners, unit, ground, prevCorners, prevHomography):
    biasCorners = []
    for x,y in corners:
        xRange = range(round(x-unit), round(x+unit))
        yRange = range(round(y-unit), round(y+unit))
        for nx,ny in it.product(xRange, yRange):
            if (nx,ny) in prevCorners: 
                biasCorners.append([x,y])
                break 

    homography = icp(biasCorners, ground, init_pose=prevHomography)
    
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
            mask (np.array)
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

def transform_update(rawImage, image, frame, vanishing, unit, corners, identities, neighborInfo, interpolationData):
    '''
    Perspective transformation via homography mapping 

    Input: image (np.array)
            identities (dict) 
    Output: image (np.array)
    '''
    return image_transform(rawImage, image, frame, vanishing, unit, corners, identities, neighborInfo, interpolationData)

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
    
def log(frame, image, homographyImage, rect_matrix, unit, identities, homography, interpolationData, save, logging, show):
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
    if logging: save_pickle(idst + 'interpolation.pickle', interpolationData)

def pipeline(frame, image, low, high, ground, imageUnit=48):
    # load image and previous data 
    rawImage = copy.deepcopy(image) 
    prevData = load_pickle(ddst + '{}.pickle'.format(frame-1), dict())

    # image rectification 
    image, mask, rect_matrix, unit, useInter = preprocess(image, frame, low, high, prevData)

    # corner detection 
    corners, minMaxPosition = corner_detection(image, mask, unit)
    featureVectors, neighborInfo = get_corner_info(corners, unit)

    # run icp algorithms
    if prevData: 
        prevHomography = prevData.get('homography', None)
        prevCorners = set(prevData.get('identities', dict()).keys())
        ground = np.array(list(pixelInfo.values()))
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
            ix, iy = transform_coord(x, y, icpHomography)
            xRange = range(round(ix-distance), round(ix+distance))
            yRange = range(round(iy-distance), round(iy+distance))
            for nx,ny in it.product(xRange, yRange):
                if nx,ny in pixelCorner: 
                    icpIdentities[(x,y)] = pixelCorner[(nx,ny)]
        
        cornerMatching = analyze_icp(icpIdentities, featureVectors, neighborInfo)
    else: 
        cornerMatching = analyze_neighbor(featureVectors, neighborInfo, minMaxPosition)
    
    image, identities, red, green, blue = analyze_region(image, unit, cornerMatching, neighborInfo)
    

# def pipeline(frame, image, low, high, interpolationData, useDeblur=False, prevInter=False, fast=False, show=False, save=True, logging=True):
#     rawImage = copy.deepcopy(image)
#     start_time = time.time() 

#     prevData = load_pickle(ddst + '{}.pickle'.format(frame-1), dict())
    
#     image, mask, vanishing, unit, useInter = preprocess(image, frame, low, high, prevData, prevInter=prevInter, fast=fast)
#     preprocess_time = time.time() 
#     print ('Preprocess Time: {}'.format(preprocess_time-start_time))

#     corners, minMaxPosition = corner_detection(image, mask, unit)
#     featureVectors, neighborInfo = get_corner_info(corners, unit)
#     cornerMatching = analyze_neighbor(featureVectors, neighborInfo, minMaxPosition)
#     image, identities, red, green, blue = analyze_region(image, unit, cornerMatching, neighborInfo)
#     identities = temporal_analysis(image, frame, unit, corners, identities)
#     purple = copy.deepcopy(identities)
#     analyze_time = time.time() 
#     print ('Analyze Time: {}'.format(analyze_time-preprocess_time))

#     image, vanishing, unit, identities, homography, status = transform_update(rawImage, image, frame, vanishing, unit, corners, identities, neighborInfo, interpolationData) 
#     everything = copy.deepcopy(identities)
#     image = sequential_label(image, red, green, blue, purple, everything)
#     homographyImage = warp_image(image, homography)
#     homography_time = time.time()
#     print ('Homography Time: {}'.format(homography_time-analyze_time))

#     if not status and not useDeblur: 
#         print ('deblur')
#         return pipeline(frame, deblur(rawImage), low, high, interpolationData, useDeblur=True, prevInter=prevInter, fast=fast, show=show, save=save, logging=logging)
    
#     log(frame, image, homographyImage, vanishing, unit, identities, homography, interpolationData, save=save, logging=logging, show=show)
#     return useInter

def main():
    frameRange = range(200, 205)
    interpolationData = load_pickle(idst + 'interpolation.pickle', set())
    prevInter = conditions['prevInter'] 

    for frame in frameRange:
        print (frame)
        fname = src + 'frame{}.png'.format(frame)
        image = cv2.imread(fname)

        if frame == frameRange[0]: 
            low, high = select_region(image)

        try: 
            prevInter = pipeline(frame, image, low, high, interpolationData, prevInter=prevInter, fast=conditions['fast'], show=conditions['show'] , save=conditions['save'] , logging=conditions['logging'])
        except: 
            print ('error in pipeline')
            image, homography, vanishing, unit, identities = interpolate(image, frame)
            homographyImage = warp_image(image, homography)
            log(frame, image, homographyImage, vanishing, unit, identities, homography, interpolationData, show=conditions['show'] , save=conditions['save'] , logging=conditions['logging'])
            prevInter = conditions['prevInter']  

main()