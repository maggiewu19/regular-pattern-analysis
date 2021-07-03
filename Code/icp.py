from utility import * 
from directory import *
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

from deblur_image import *
from rectify_image import *
from create_mask import *
from detect_feature import * 
from find_homography import * 
from post_processing import *

def create_bipartite(A, B): 
    '''
    A - image 
    B - reference 
    '''
    bipartite = []
    reference = [[bx,by] for bx,by in B]
    image = [[ax,ay] for ax,ay in A]

    for ax,ay in A: 
        adjacency = [] 
        for bx,by in B: 
            distance = euclidean(ax, ay, bx, by)
            adjacency.append(distance) 
        bipartite.append(adjacency)
    
    return np.array(bipartite), np.array(image), np.array(reference)

def best_match(bipartite, image, reference, ratio):
    rows, cols = linear_sum_assignment(bipartite) 
    cost = np.argsort(bipartite[rows, cols])
    include = np.sort(cost)[int(ratio*len(cost))-1]
    order = cost[cost <= include]
    new_rows = rows[order]
    new_cols = cols[order]
    distances = bipartite[new_rows, new_cols].sum() / len(bipartite[new_rows, new_cols])

    # return distances, image[rows, :], reference[cols, :]
    return distances, image[new_rows, :], reference[new_cols, :]

def compute(src, dst, ratio=1, m=2):
    bipartite, image, reference = create_bipartite(src[:m,:].T, dst[:m,:].T)
    distances, selected_image, selected_reference = best_match(bipartite, image, reference, ratio)
    T, _ = cv2.findHomography(selected_image, selected_reference, method=cv2.LMEDS)
    
    return T, distances

def get_corners(corners, T, m=2):
    output = np.ones((len(corners), m+1))
    output[:,0:m] = np.copy(corners)

    output = np.dot(T, output.T).T

    return output[:,0:m]

def icp(A, B, init_pose=None, max_iterations=20, tolerance=1e-2):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    optimal = {'best_dist': float('inf'), 'target': src}

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    else:
        T, distances = compute(src, dst)
        src = np.dot(T, src)

    # pt_color = np.array([0, 0, 0])
    prev_error = 0
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        T, distances = compute(src, dst)
        src = np.dot(T, src)
        if distances < optimal['best_dist']:
            optimal['best_dist'] = distances 
            optimal['target'] = copy.deepcopy(src)

        homography, _ = cv2.findHomography(A, src[:m,:].T, method=cv2.LMEDS)
        intermediate = get_corners(A, homography)

        # cv2.imwrite(f'Sequence/Original/{i}.jpg', cv2.warpPerspective(frame, homography, (825, 1275)))

        # for x,y in intermediate: 
        #     cv2.circle(img, (int(round(x)), int(round(y))), radius=3, thickness=3, color=pt_color.tolist())

        # print (distances)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance: break

        prev_error = mean_error
        # pt_color += np.array([10,10,10])

    # calculate final transformation
    homography, _ = cv2.findHomography(A, optimal['target'][:m,:].T, method=cv2.LMEDS)

    return homography

def run_icp(corners, ground, img, frame):
    T = icp(corners, ground, img, frame)
    return get_corners(corners, T)


# epsilon=1e-4
# k=5e-2
# block=5
# original = cv2.imread("Data/Lee/Frames/frame200.png")
# low, high = select_region(original)
# mask = get_mask(original, low, high)
# gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# features = cv2.goodFeaturesToTrack(gray, maxCorners=200, 
#                             qualityLevel=epsilon, minDistance=10, 
#                             useHarrisDetector=True, blockSize=block, k=k, mask=mask)

# original_corners = []
# for corner in np.int64(features): 
#     x,y = corner.ravel()
#     original_corners.append([int(x),int(y)])
#     cv2.circle(original, (int(round(x)), int(round(y))), radius=1, thickness=1, color=(0,0,0))

# original_corners = np.array(original_corners)

# corners = np.load("corners.npy")
# all_corners = np.load("200corners.npy")
# corresponding_corner = np.load("correspondence.npy")
# # ground = np.array([pixelInfo[corner] for corner in corresponding_corner])
# ground = np.array(list(pixelInfo.values()))

# base = cv2.imread("Maze_info/orthogonal_view.png")
# icp_base = cv2.imread("Maze_info/orthogonal_view.png")
# frame = cv2.imread("Data/Lee/Corners/200.jpg")

# # homography, _ = cv2.findHomography(corners, ground, method=cv2.LMEDS)
# output = run_icp(all_corners, ground, icp_base, frame)

# # for x,y in output: 
# #     cv2.circle(icp_base, (int(round(x)), int(round(y))), radius=3, thickness=3, color=(0,0,0))

# # for x,y in corners: 
# #     new_x, new_y = transform_coord(x, y, homography)
# #     cv2.circle(base, (int(round(new_x)), int(round(new_y))), radius=3, thickness=3, color=(0,0,0))

# # cv2.imshow("base", base)
# cv2.imshow("icp base", icp_base)
# cv2.waitKey(0)
