from directory import *
from utility import *

### Parameters #### 
dthreshold = 50.0 
ethreshold = 0.05
tthreshold = np.pi*5/180
disector = [0, -1, -1, -1, 0, 1, 1, 1]
djsector = [1, 1, 0, -1, -1, -1, 0, 1]
dwindow = 7
nangle = 360*5 
nlinesmax = 128 
hthreshold = 15000.0
halfsizeflag = True 
spreadcontribution = False 

def derivatives(image, i, j):
    dx = (image[i-1][j+1] + 2 * image[i][j+1] + image[i+1][j+1]) - (image[i-1][j-1] + 2 * image[i][j-1] + image[i+1][j-1])
    dy = (image[i-1][j-1] + 2 * image[i-1][j] + image[i-1][j+1]) - (image[i+1][j-1] + 2 * image[i+1][j] + image[i+1][j+1])
    dxx = (image[i-1][j-1] + 2 * image[i][j-1] + image[i+1][j-1]) - 2*(image[i-1][j] + 2 * image[i][j] + image[i+1][j]) + (image[i-1][j+1] + 2*image[i][j+1] + image[i+1][j+1])
    dxy = (image[i+1][j-1] - image[i-1][j-1] - image[i+1][j+1] + image[i-1][j+1])
    dyy = (image[i-1][j-1] + 2 * image[i-1][j] + image[i-1][j+1]) - 2*(image[i][j-1] + 2 * image[i][j] + image[i][j+1]) + (image[i+1][j-1] + 2*image[i+1][j] + image[i+1][j+1])

    return dx, dy, dxx, dxy, dyy 

def find_sector(cost, sint):
    sin22d5 = np.sin(np.pi/8)

    if cost >= 0: 
        if sint >= 0: 
            if sint < sin22d5: return 0 
            elif cost > sin22d5: return 1 
            else: return 2 
        else: 
            if sint > -sin22d5: return 0 
            elif cost > sin22d5: return 7 
            else: return 6 
    else: 
        if sint >= 0: 
            if sint < sin22d5: return 4
            elif cost < -sin22d5: return 3
            else: return 2 
        else: 
            if sint > -sin22d5: return 4
            elif cost < -sin22d5: return 5
            else: return 6 

def setup_tables(nangle):
    if halfsizeflag: nanglemax = nangle//2
    else: nanglemax = nangle

    costable = [] 
    sintable = []

    for i in range(nanglemax):
        alpha = 2*np.pi * i / nangle 
        costable.append(np.cos(alpha))
        sintable.append(np.sin(alpha))

    return costable, sintable 

def accumulate_hough(hough, nangle, costable, sintable, hoffsets, di, dj, wa, cost, sint):
    if halfsizeflag: nanglemax = nangle//2
    else: nanglemax = nangle

    for i in range(nanglemax): 
        cosa = costable[i]
        sina = sintable[i]
        offset = -di*cosa - dj*sina 
        j = int(offset + hoffsets/2 + 0.5)
        sinat = (cost*sina - sint*cosa)
        wg = sinat * sinat 
        w = wa * wg 
        if j >= 0 and j < hoffsets: hough[i][j] += w 
        else: print ('ERROR: hough error')

        if spreadcontribution: 
            for dii in range(-1, 2): 
                for djj in range(-1, 2):
                    if i + dii >= 0 and i + dii < nanglemax and j + djj >= 0 and j + djj < hoffsets: 
                        if dii == 0 and djj == 0: continue 
                        elif dii == 0 or djj == 0: hough[i+dii][j+djj] += w/2 
                        else: hough[i+dii][j+djj] += w/4 

def find_localpeak(hough, nangle, hoffsets, i, j): 
    dr = (dwindow-1) // 2

    for di in range(-dr, dr+1): 
        for dj in range(-dr, dr+1): 
            if di == 0 and dj == 0: continue 
            ii = i + di 
            jj = j + dj 

            if not halfsizeflag: 
                if ii < 0: ii += nangle
            else: 
                if ii < 0: 
                    ii += nangle//2 
                    jj = hoffsets-1 - jj 
            
            if not halfsizeflag: 
                if ii > nangle-1: ii -= nangle 
            else: 
                if ii > nangle//2-1: 
                    ii -= nangle//2 
                    jj = hoffsets-1 - jj 
            
            if jj < 0: jj = 0 
            if jj > hoffsets-1: jj = hoffsets-1 

            if hough[i][j] < hough[ii][jj]: return 0 
    
    return 1 

def store_hough(hough, nangle, hoffsets, lines):
    count = 0 
    if halfsizeflag: nanglemax = nangle//2
    else: nanglemax = nangle

    for i in range(nanglemax):
        for j in range(hoffsets):
            if hough[i][j] > hthreshold: 
                if find_localpeak(hough, nangle, hoffsets, i, j): 
                    offset = j - hoffsets/2 
                    alpha = 2*np.pi * i / nangle 
                    lines[count][0] = alpha
                    lines[count][1] = offset 
                    lines[count][2] = hough[i][j] 
                    count += 1 
                    if count >= nlinesmax-1: return -1 
    
    lines[count][0] = 0
    lines[count][1] = 0 
    lines[count][2] = 0 

    return count 

def angle_fit(dismiss, thetas):
    more_dismiss = False 
    theta_med = np.median(list(thetas.values()))
    thetas_copy = thetas.copy()

    for i in thetas_copy:
        if i in dismiss: del thetas[i]

    for i in thetas: 
        if min(np.pi-abs(thetas[i]-theta_med), abs(thetas[i]-theta_med)) > tthreshold:
            if i not in dismiss: 
                dismiss.add(i)
                more_dismiss = True 
    
    return more_dismiss 

def error_fit(dismiss, infos, error, distsq, x, y):
    ind_error = dict()
    more_dismiss = False 
    infos_copy = infos.copy()

    for i in infos_copy:
        if i in dismiss: del infos[i]

    for i in infos: 
        cost, sint, rho = infos[i] 
        ind_error[i] = abs((x*sint - y*cost + rho) / distsq)

    if len(ind_error) == 0: 
        return more_dismiss
        
    if error >= ethreshold: 
        i = max(ind_error.items(), key=operator.itemgetter(1))[0]
        if i not in dismiss: 
            dismiss.add(i)
            more_dismiss = True 
    
    return more_dismiss

def intersection_fit(dismiss, intersections, flag, height, width, ratio=1):
    count_inter = dict()
    more_dismiss = False 
    intersections_copy = intersections.copy()

    for i in intersections_copy:
        if i in dismiss: del intersections[i]

    io = height/2 
    jo = width/2 

    for i in intersections: 
        for j in intersections: 
            if i != j: 
                theta1, rho1, wi1 = intersections[i]
                theta2, rho2, wi2 = intersections[j]
                lines = [intersections[i], intersections[j]]

                x, y, _, _, _ = least_squares_fit(lines, 2, flag, set(), height, width, test=False)

                if jo-ratio*width <= x+jo < jo+ratio*width and io-ratio*height <= io-y < io+ratio*height: 
                    count_inter[i] = count_inter.get(i,0) + 1 
                    count_inter[j] = count_inter.get(j,0) + 1 

    if len(count_inter) > 0: 
        i, count = max(count_inter.items(), key=operator.itemgetter(1))
        if count > 2: 
            dismiss.add(i)
            more_dismiss = True 

    return more_dismiss

def center_fit(dismiss, infos, flag, height, width, ratio=0.8):
    if flag: return False 

    more_dismiss = False 
    infos_copy = infos.copy()

    for i in infos_copy:
        if i in dismiss: del infos[i]

    for i in infos: 
        _, _, rho = infos[i]
        if abs(rho) > 0.5*ratio*width:
            if i not in dismiss: 
                dismiss.add(i)
                more_dismiss = True 
    
    return more_dismiss 

def least_squares_fit(lines, n, flag, dismiss, height, width, aiaisum=0, aibisum=0, bibisum=0, aicisum=0, bicisum=0, cicisum=0, wsum=0, test=True):
    infos = dict()
    thetas = dict()
    intersections = dict()

    io = height/2 
    jo = width/2 
    
    if n < 2: return -1 

    count = 0 
    for i in range(n): 
        theta = lines[i][0]
        cost = np.cos(theta)
        sint = np.sin(theta)
        rho = lines[i][1]
        wi = lines[i][2]

        valid = (abs(sint) < abs(cost)) == flag
        
        if valid and i not in dismiss: 
            ai = sint 
            bi = -cost 
            ci = rho 

            aiaisum += ai * ai * wi 
            aibisum += ai * bi * wi 
            bibisum += bi * bi * wi 
            aicisum += ai * ci * wi 
            bicisum += bi * ci * wi 
            cicisum += ci * ci * wi 
            wsum += wi 
            count += 1 

            thetas[i] = theta 
            infos[i] = [cost, sint, rho]
            intersections[i] = [theta, rho, wi]
    
    det = aiaisum * bibisum - aibisum * aibisum

    # Avoid dividion by zero error
    if abs(det) < 1e-8: 
        x = float('inf')
        y = float('inf')
        distsq = float('inf')
        errsq = float('inf')
        error = float('inf')
    else: 
        x = (aibisum * bicisum - bibisum * aicisum) / det 
        y = (aibisum * aicisum - aiaisum * bicisum) / det 
        distsq = np.sqrt(round((io-y)**2 + (jo-x)**2))
        errsq = round((x * x * aiaisum + 2 * x * y * aibisum + y * y * bibisum + cicisum + 2 * x * aicisum + 2 * y * bicisum) / wsum)
        error = np.sqrt(errsq) / distsq 

    if test: 
        # Angle Test / Error Test / Intersection Test

        angle_test = angle_fit(dismiss, thetas)
        error_test = error_fit(dismiss, infos, error, distsq, x, y)
        intersection_test = intersection_fit(dismiss, intersections, flag, height, width)
        center_test = center_fit(dismiss, infos, flag, height, width)

        if angle_test or error_test or intersection_test or center_test: 
            return least_squares_fit(lines, n, flag, dismiss, height, width)

    return x, y, count, error, dismiss 

def draw_lines(image, lines, n, hdismiss, vdismiss):
    output = copy.copy(image)
    height, width, depth = output.shape 

    io = height/2 
    jo = width/2 

    for i in range(n): 
        theta = lines[i][0]
        cost = np.cos(theta)
        sint = np.sin(theta)
        rho = lines[i][1]

        if i in hdismiss.union(vdismiss): color = (255,0,0)
        elif abs(sint) < abs(cost): color = (0,0,255)
        else: color = (0,255,0)

        if abs(sint) < abs(cost): 
            for j in range(width):
                dj = j-jo 
                i = int(io - (rho + dj * sint)/cost + 0.5)
                if (i >= 0 and i < height): 
                    cv2.circle(output, (j, i), 1, color)
        else: 
            for i in range(height):
                di = io-i 
                j = int(jo - (rho - di * cost)/sint + 0.5)
                if (j >= 0 and j < width):
                    cv2.circle(output, (j, i), 1, color)
    
    return output

def extract_unit(lines, n, flag, hdismiss, vdismiss):
    rhos = list()
    diffs = list()
    for i in range(n):
        theta = lines[i][0]
        cost = np.cos(theta)
        sint = np.sin(theta)
        rho = lines[i][1]
        wi = lines[i][2]

        valid = (abs(sint) < abs(cost)) == flag

        if valid and i not in hdismiss.union(vdismiss):
            rhos.append(rho)
        
    rhos.sort()
    for i in range(len(rhos)-1): 
        diffs.append(rhos[i+1]-rhos[i])
    
    return np.median(diffs)

def extract_lines(image, lines):
    height, width = image.shape 
    count = 0
    io = height/2 
    jo = width/2 
    if halfsizeflag: nanglemax = nangle//2
    else: nanglemax = nangle

    hoffsets = 2*int(np.sqrt(height*height + width*width)+0.5)+1
    hough = np.zeros((nanglemax, hoffsets))
    output = np.zeros((height, width))
    costable, sintable = setup_tables(nangle)

    for i in range(1, height-1):
        for j in range(1, width-1): 
            dx, dy, dxx, dxy, dyy = derivatives(image, i, j)
            d = np.sqrt((dxx-dyy) * (dxx-dyy) + 4 * dxy * dxy)

            if d <= dthreshold: 
                output[i][j] = 0 
                continue 
                
            if dxx + dyy >= 0: 
                cos2t = (dxx - dyy) / d 
                sin2t = 2 * dxy / d 
                curv = (d + (dxx + dyy)) / 2 
            else: 
                cos2t = -(dxx - dyy) / d 
                sin2t = -2 * dxy / d 
                curve = (d - (dxx + dyy)) / 2

            sint = np.sqrt((1-cos2t)/2)
            if sint != 0: cost = sin2t / (2*sint)
            else: cost = np.sqrt((1+cos2t)/2)

            sector = find_sector(cost, sint)

            ia = i - disector[sector]
            ja = j - djsector[sector]
            ic = i + disector[sector]
            jc = j + djsector[sector]
            ib = i 
            jb = j

            a = image[ia][ja]
            b = image[ib][jb]
            c = image[ic][jc]

            if b <= a and b < c: 
                s = 0 
                count += 1 
                val = min(4 * (a+c - 2*b), 255)
                
                if a+c - 2*b != 0: s = (a-c) / (2*(a+c - 2*b))
                else: print ('ERROR: s = 0')
                
                if s < -0.5 or s > 0.5: print ('ERROR: s value incorrect')
                
                if s >= 0: 
                    output[ic][jc] = val * s 
                    output[ib][jb] = val * (1-s)
                else: 
                    output[ia][ja] = val * -s 
                    output[ib][jb] = val * (1+s)
                
                di = s * (ic-ia) / 2
                dj = s * (jc-ja) / 2

                dproj = dj*cost - di*sint 

                did = -dproj * sint 
                djd = dproj * cost 

                accumulate_hough(hough, nangle, costable, sintable, hoffsets, (i+did)-io, (j+djd)-jo, val, cost, sint)
            else: 
                output[ib][jb] = 0

    nlines = store_hough(hough, nangle, hoffsets, lines)

    return lines, nlines 

def find_vanishing(image, lines, n):
    height, width, depth = image.shape 

    phx, phy, phnused, pherr, hdismiss = least_squares_fit(lines, n, 1, set(), height, width)
    hy = height/2 - phy
    hx = width/2 + phx

    pvx, pvy, pvnused, pverr, vdismiss = least_squares_fit(lines, n, 0, set(), height, width)
    vy = height/2 - pvy
    vx = width/2 + pvx

    return hx, hy, vx, vy, hdismiss, vdismiss

def compute_matrices(hx, hy, vx, vy, height, width, camera_matrix):
    inverse_matrix = np.linalg.inv(camera_matrix)
    px = camera_matrix[0][2]
    py = height - camera_matrix[1][2] 

    vanish = np.array([[hx, height-hy, 1.0], [vx, height-vy, 1.0]])
    axes = sp.normalize(inverse_matrix.dot(np.transpose(vanish)), axis=0).T 
    if axes[0][0] < 0: axes[0,:] = -axes[0,:]
    if axes[1][1] < 0: axes[1,:] = -axes[1,:]
    axes = np.vstack((axes, np.cross(axes[0,:], axes[1,:]).reshape(1,3)))
    if axes[2][2] < 0: print ('ERROR: cross product < 0')
    
    rotation_matrix = np.transpose(axes)

    avec = np.array([0, 0, 1]).reshape((3,1))
    bvec = np.dot(axes, avec).reshape((3,1))
    bvec /= bvec[2]
    cvec = np.dot(camera_matrix, bvec).reshape((3,1))
    dx = cvec[0][0] - px
    dy = py - cvec[1][0]

    return inverse_matrix, rotation_matrix, dx, dy 

def rectify(image, hx, hy, vx, vy): 
    height, width, depth = image.shape
    dim = (width, height)
    inverse_matrix, rotation_matrix, dx, dy = compute_matrices(hx, hy, vx, vy, height, width, camera_matrix)

    correction = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height], [0.0, 0.0, 1.0]])
    output = cv2.warpPerspective(image, correction, dim, flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)

    vector = np.array([[1.0, 0.0, dx], [0.0, -1.0, height-dy], [0.0, 0.0, 1.0]])
    perspective = np.dot(camera_matrix, np.dot(rotation_matrix, np.dot(inverse_matrix, vector)))
    output = cv2.warpPerspective(output, perspective, dim, flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)

    return output

def find_lines(image):
    height, width, depth = image.shape 
    b_in, g_in, r_in = cv2.split(image)

    lines = np.zeros((nlinesmax, 3))

    # bhx, bhy, bvx, bvy = extract_lines(b_in, lines)
    lines, n = extract_lines(g_in, lines)
    # rhx, rhy, rvx, rvy = extract_lines(r_in, lines)

    return lines, n

def find_unit(lines, n, hdismiss, vdismiss):
    hunit = extract_unit(lines, n, 1, hdismiss, vdismiss)
    vunit = extract_unit(lines, n, 0, hdismiss, vdismiss)

    return int(min(hunit, vunit))