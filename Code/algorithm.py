from rectify_image import *
from create_mask import *
from detect_feature import * 
from find_homography import * 
from post_processing import *

root = '/Users/maggiewu/Documents/Post_MEng_Research/'
src = root + 'Data/Frames/'
cdst = root + 'Data/Corners/'
hdst = root + 'Data/Homography/'
ddst = root + 'Logging/'

def preprocess(image, frame, low, high, frame_data):
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
    try: 
        lines, n = find_lines(image.astype(np.int16))
        hx, hy, vx, vy, hdismiss, vdismiss = find_vanishing(image, lines, n)
        unit = find_unit(lines, n, hdismiss, vdismiss)
    except: 
        print ('frame {} requires interpolation'.format(frame))
        hx, hy, vx, vy = frame_data['vanishing']
        unit = frame_data['unit']

    image = rectify(image, hx, hy, vx, vy)
    mask = get_mask(image, low, high)

    frame_data['vanishing'] = [hx, hy, vx, vy]
    frame_data['unit'] = unit 

    return image, mask, unit 

def analyze_region(image, mask, unit, maxCorners=300, epsilon=1e-4, k=5e-2, block=5):
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
    while moreIdentities: 
        identities, takenCorners, moreIdentities = extend_identities(image, neighborInfo, identities, takenCorners)
        distance_check(unit, identities, takenCorners) 
        cornerLabels = label_corners(image, identities, (0,150,0), cornerLabels=cornerLabels)
    
    moreIdentities = True 
    while moreIdentities: 
        identities, takenCorners, moreIdentities = extend_identities(image, neighborInfo, identities, takenCorners, minScore=1)
        distance_check(unit, identities, takenCorners) 
        cornerLabels = label_corners(image, identities, (255,0,0), cornerLabels=cornerLabels)

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

def pipeline(frame, image, low, high, frame_data, show=False, save=True, logging=True):
    image, mask, unit = preprocess(image, low, high, frame_data)
    image, corners, identities, cornerLabels = analyze_region(image, mask, unit)
    identities, cornerLabels = temporal_analysis(image, unit, corners, identities, cornerLabels, frame_data)
    
    if save: cv2.imwrite(cdst + '{}.jpg'.format(frame), image)

    homography = get_homography(identities)
    identities, cornerLabels = template_match(image, unit, corners, identities, cornerLabels, homography)
    image, homography, meanDist, errorCount = transform(image, identities)

    frame_data['identities'] = identities 
    frame_data['homography'] = homography 

    if show: 
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    if save: cv2.imwrite(hdst + '{}.jpg'.format(frame), image)
    if logging: save_pickle(ddst + '{}.pickle'.format(frame), frame_data)

def main():
    frameRange = range(0, 920)
    frame_data = {'vanishing': None, 'unit': None, 'identities': dict(), 'homography': None}

    for frame in frameRange:
        fname = src + 'frame{}.png'.format(frame)
        image = cv2.imread(fname)

        if frame == frameRange[0]: 
            low, high = select_region(image)

        pipeline(frame, image, low, high, frame_data)

main()