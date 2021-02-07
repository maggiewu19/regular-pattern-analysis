from directory import *
from deblur_image import *
from rectify_image import *
from create_mask import *
from detect_feature import * 
from find_homography import * 
from post_processing import *

import time 

def preprocess(image, frame, low, high, prevInter=False, fast=False, interval=5):
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
    prevData = load_pickle(ddst + '{}.pickle'.format(frame-1), dict())
    
    def run():
        try: 
            lines, n = find_lines(image.astype(np.int16))
            hx, hy, vx, vy, hdismiss, vdismiss = find_vanishing(image, lines, n)
            unit = find_unit(lines, n, hdismiss, vdismiss)
            prevInter = False 
        except: 
            print ('frame {} requires rectification interpolation'.format(frame))
            hx, hy, vx, vy = prevData['vanishing']
            unit = prevData['unit']
            prevInter = True 
        
        return hx, hy, vx, vy, unit, prevInter 

    if not fast: 
        hx, hy, vx, vy, unit, prevInter = run()
    else: 
        if frame % interval == 0 or prevInter: 
            hx, hy, vx, vy, unit, prevInter = run()
        else: 
            hx, hy, vx, vy = prevData['vanishing']
            unit = prevData['unit']
            prevInter = False 

    vanishing = [hx, hy, vx, vy]
    image = rectify(image, hx, hy, vx, vy)
    mask = get_mask(image, low, high)

    return image, mask, vanishing, unit, prevInter 

def analyze_region(image, mask, unit):
    '''
    Corner detection via Harris corner detector 
    Corner matching and regional pattern analysis 

    Input: image (np.array)
            mask (np.array)
            unit (float)
    Output: image (np.array)
            identities (dict) 
    '''
    corners, minMaxPosition = corner_detection(image, mask, unit)
    featureVectors, neighborInfo = get_corner_info(corners, unit)
    cornerMatching = get_corner_matching(featureVectors, neighborInfo)
    cornerMatching = section_split(minMaxPosition, cornerMatching)
    cornerMatching = identify_matches(neighborInfo, cornerMatching)

    identities, takenCorners = assign_identities(image, cornerMatching) 
    distance_check(unit, identities, takenCorners) 
    red = copy.deepcopy(identities)
    
    identities, takenCorners = iterative_extend(image, unit, neighborInfo, identities, takenCorners)
    green = copy.deepcopy(identities)
    
    identities, takenCorners = iterative_extend(image, unit, neighborInfo, identities, takenCorners, minScore=1)
    blue = copy.deepcopy(identities)

    return image, corners, identities, neighborInfo, red, green, blue

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
            corner = target[(x,y)]
            if (x,y) in everything and (x,y) not in labelled: 
                labelled.add((x,y))
                cv2.putText(image, str(corner), org=(x-3*len(str(corner)), y-3), fontFace=cv2.FONT_HERSHEY_PLAIN, color=color, fontScale=fontScale) 
        return labelled 

    labelled = label(red, (0,0,255), set())
    labelled = label(green, (0,150,0), labelled)
    labelled = label(blue, (255,0,0), labelled)
    labelled = label(purple, (150,0,150), labelled)
    labelled = label(everything, (150,150,0), labelled)

    return image 
    
def log(frame, image, homographyImage, vanishing, unit, identities, homography, interpolationData, save, logging, show):
    frameData = {'vanishing': None, 'unit': None, 'identities': dict(), 'homography': None}
    frameData['vanishing'] = vanishing
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

def pipeline(frame, image, low, high, interpolationData, useDeblur=False, prevInter=False, fast=False, show=False, save=True, logging=True):
    rawImage = copy.deepcopy(image)
    start_time = time.time() 
    
    image, mask, vanishing, unit, useInter = preprocess(image, frame, low, high, prevInter=prevInter, fast=fast)
    preprocess_time = time.time() 
    print ('Preprocess Time: {}'.format(preprocess_time-start_time))

    image, corners, identities, neighborInfo, red, green, blue = analyze_region(image, mask, unit)
    identities = temporal_analysis(image, frame, unit, corners, identities)
    purple = copy.deepcopy(identities)
    analyze_time = time.time() 
    print ('Analyze Time: {}'.format(analyze_time-preprocess_time))

    image, vanishing, unit, identities, homography, status = transform_update(rawImage, image, frame, vanishing, unit, corners, identities, neighborInfo, interpolationData) 
    everything = copy.deepcopy(identities)
    image = sequential_label(image, red, green, blue, purple, everything)
    homographyImage = warp_image(image, homography)
    homography_time = time.time()
    print ('Homography Time: {}'.format(homography_time-analyze_time))

    if not status and not useDeblur: 
        print ('deblur')
        return pipeline(frame, deblur(rawImage), low, high, interpolationData, useDeblur=True, prevInter=prevInter, fast=fast, show=show, save=save, logging=logging)
    else: 
        log(frame, image, homographyImage, vanishing, unit, identities, homography, interpolationData, save=save, logging=logging, show=show)

    return useInter

def main():
    frameRange = range(330, 335)
    interpolationData = load_pickle(idst + 'interpolation.pickle', set())
    prevInter = True 

    for frame in frameRange:
        print (frame)
        fname = src + 'frame{}.png'.format(frame)
        image = cv2.imread(fname)

        if frame == frameRange[0]: 
            low, high = select_region(image)

        try: 
            prevInter = pipeline(frame, image, low, high, interpolationData, prevInter=prevInter, fast=True)
        except: 
            print ('error in pipeline')
            image, homography, vanishing, unit, identities = interpolate(image, frame)
            homographyImage = warp_image(image, homography)
            log(frame, image, homographyImage, vanishing, unit, identities, homography, interpolationData, show=False, save=True, logging=True)
            prevInter = True 

main()