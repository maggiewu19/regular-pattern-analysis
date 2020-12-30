from rectify_image import *
from create_mask import *
from detect_feature import * 
from find_homography import * 
from post_processing import *

import time 

root = '/Users/maggiewu/Documents/Post_MEng_Research/'
src = root + 'Data/Frames/'
cdst = root + 'Data/Corners/'
hdst = root + 'Data/Homography/'
ddst = root + 'Logging/'

def preprocess(image, frame, low, high, frame_data, prev_inter=False, fast=False):
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
    if not fast: 
        try: 
            lines, n = find_lines(image.astype(np.int16))
            hx, hy, vx, vy, hdismiss, vdismiss = find_vanishing(image, lines, n)
            unit = find_unit(lines, n, hdismiss, vdismiss)
            prev_inter = False 
        except: 
            print ('frame {} requires interpolation'.format(frame))
            hx, hy, vx, vy = frame_data['vanishing']
            unit = frame_data['unit']
            prev_inter = True 
    else: 
        if frame % 10 == 0 or prev_inter: 
            try: 
                lines, n = find_lines(image.astype(np.int16))
                hx, hy, vx, vy, hdismiss, vdismiss = find_vanishing(image, lines, n)
                unit = find_unit(lines, n, hdismiss, vdismiss)
                prev_inter = False 
            except: 
                print ('frame {} requires interpolation'.format(frame))
                hx, hy, vx, vy = frame_data['vanishing']
                unit = frame_data['unit']
                prev_inter = True 
        else: 
            hx, hy, vx, vy = frame_data['vanishing']
            unit = frame_data['unit']
            prev_inter = False 

    image = rectify(image, hx, hy, vx, vy)
    mask = get_mask(image, low, high)

    frame_data['vanishing'] = [hx, hy, vx, vy]
    frame_data['unit'] = unit 

    return image, mask, unit, prev_inter 

def analyze_region(image, mask, unit, maxCorners=300, epsilon=1e-4, k=5e-2, block=5, maxCount=20):
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

    identities, takenCorners, _ = assign_identities(image, cornerMatching) 
    distance_check(unit, identities, takenCorners) 
    cornerLabels = label_corners(image, identities, (0,0,255))
    
    moreIdentities = True 
    count = 0
    while moreIdentities and count < maxCount: 
        identities, takenCorners, moreIdentities = extend_identities(image, neighborInfo, identities, takenCorners)
        distance_check(unit, identities, takenCorners) 
        cornerLabels = label_corners(image, identities, (0,150,0), cornerLabels=cornerLabels)
        count += 1 
    
    moreIdentities = True
    count = 0 
    while moreIdentities and count < maxCount: 
        identities, takenCorners, moreIdentities = extend_identities(image, neighborInfo, identities, takenCorners, minScore=1)
        distance_check(unit, identities, takenCorners) 
        cornerLabels = label_corners(image, identities, (255,0,0), cornerLabels=cornerLabels)
        count += 1 

    return image, corners, identities, cornerLabels 

def transform(image, identities):
    '''
    Perspective transformation via homography mapping 

    Input: image (np.array)
            identities (dict) 
    Output: image (np.array)
    '''
    homography = get_homography(identities)
    image = warp_image(image, homography)
    meanDist, errorCount = image_error(identities, homography)

    return image, homography, meanDist, errorCount

def pipeline(frame, image, low, high, frame_data, prev_inter=False, fast=False, show=False, save=True, logging=True):
    start_time = time.time() 
    image, mask, unit, prev_inter = preprocess(image, frame, low, high, frame_data, prev_inter=prev_inter, fast=fast)
    preprocess_time = time.time() 
    print ('Preprocess Time: {}'.format(preprocess_time-start_time))
    image, corners, identities, cornerLabels = analyze_region(image, mask, unit)
    analyze_time = time.time() 
    print ('Analyze Time: {}'.format(analyze_time-preprocess_time))
    identities, cornerLabels = temporal_analysis(image, unit, corners, identities, cornerLabels, frame_data)
    
    if save: cv2.imwrite(cdst + '{}.jpg'.format(frame), image)

    homography = get_homography(identities)
    identities, cornerLabels = template_match(image, unit, corners, identities, cornerLabels, homography)
    image, homography, meanDist, errorCount = transform(image, identities)
    homography_time = time.time() 
    print ('Homography Time: {}'.format(homography_time-analyze_time))

    frame_data['identities'] = identities 
    frame_data['homography'] = homography 

    if show: 
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    if save: cv2.imwrite(hdst + '{}.jpg'.format(frame), image)
    if logging: save_pickle(ddst + '{}.pickle'.format(frame), frame_data)

    return prev_inter

def main():
    frameRange = range(40, 920)
    frame_data = {'vanishing': None, 'unit': None, 'identities': dict(), 'homography': None}
    prev_inter = True 
    
    for frame in frameRange:
        print (frame)
        fname = src + 'frame{}.png'.format(frame)
        image = cv2.imread(fname)

        if frame == frameRange[0]: 
            low, high = select_region(image)

        prev_inter = pipeline(frame, image, low, high, frame_data, prev_inter=prev_inter, fast=True)

main()