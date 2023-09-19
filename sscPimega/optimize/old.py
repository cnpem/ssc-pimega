def annotation_ux_restoration( image, tracking, L, typedet, *args):    

    if typedet == "planar":

        if not args:
            xdet = get_project_values_geometry( {'geo':'planar',  'opt':False, 'mode': 'real'} ) 
        else:
            xdet = args[0]

        det = _extract_detector_dictionary_( xdet, L, {'geo':'planar','opt':False, 'mode':'real' } )
    
        vmatcho, vmatchd = tracking540D_planar(det, tracking )
    
        geo = geometry540D( det )

        inv = backward540D( image, geo )
    
        plt.figure(figsize=(8,8))
        plt.imshow(inv, cmap="jet")

        for k in range( vmatcho.shape[0] ):
            plt.plot( [int(vmatcho[k][0]) , int(vmatcho[k][2])], [int(vmatcho[k][1]), int(vmatcho[k][3])], 'r-')
        
        #plt.xlim( [1536-roi, 1536+roi] )
        #plt.ylim( [1536-roi, 1536+roi] )
        
        plt.show()

    elif typedet == "nonplanar":

        if not args:
            xdet = get_project_values_geometry( {'geo':'nonplanar',  'opt':False, 'mode': 'virtual'} ) 
        else:
            xdet = args[0]

        det = _extract_detector_dictionary_( xdet, L, {'geo':'nonplanar','opt':False, 'mode':'virtual' } )
    
        vmatcho, vmatchd = tracking540D(det, tracking )
    
        geo = geometry540D( det )

        inv = backward540D( image, geo )
    
        plt.figure(figsize=(8,8))
        plt.imshow(inv, cmap="jet")

        for k in range( vmatcho.shape[0] ):
            plt.plot( [int(vmatcho[k][0]) , int(vmatcho[k][2])], [int(vmatcho[k][1]), int(vmatcho[k][3])], 'r-')
        
        #plt.xlim( [1536-roi, 1536+roi] )
        #plt.ylim( [1536-roi, 1536+roi] )
        
        plt.show()

        


def _worker_LNLS_template_with_squares_( img,  *args ):
    #Tailored code for a measured image using a template from LNLS
    #consisting of a set of squares, each intersecting four chips
    #

    #susp = 1 
    #img  = set_suspicious_pixels_540D( img, 256, 6, 6, susp )
    
    if not args:
        IGNORE = False
        CHIPS_TO_IGNORE_ANNOTATION = []
    else:
        IGNORE = True
        CHIPS_TO_IGNORE_ANNOTATION = args[0]
        if not CHIPS_TO_IGNORE_ANNOTATION:
            IGNORE = False
         
    x = numpy.zeros([2, 24,12], dtype=int)  # x[0] top pixels / x[1] bottom pixels, for each strip
    y = numpy.zeros([2, 24,12], dtype=int)  # same, for y

    #xb - x boundary : xb[0] top , xb[1] bottom
    #yb - y boundry : yb[0] top , yb[1] bottom
    #xs - x side: xs[0] left , xs[1] right
    #ys - y side: ys[0] left , ys[1] right
    xb = numpy.zeros([2, 24, 12], dtype=int)  
    yb = numpy.zeros([2, 24, 12], dtype=int)
    xs = numpy.zeros([2, 24, 12], dtype=int)
    ys = numpy.zeros([2, 24, 12], dtype=int)
    
    def getcorner(h):
        avg = numpy.average(h)
        x = 1
        while x < 127 and ((h[x] > avg and h[x-1] > avg) or (h[x] < avg and h[x-1] < avg)):
            x += 1
        if h[x] < avg and h[x-1] > avg:
            x -= 1
        return x
    
    def getxy(chip):
        avg = numpy.average(chip)
        chip = (chip > avg)*1
        hx = numpy.sum(chip,0)
        hy = numpy.sum(chip,1)
        cx = getcorner(hx)
        cy = getcorner(hy)
        return cx,cy
    
   
    def _getxy_( quarter, side ):

        #remove nan
        #quarter[ numpy.isnan(quarter) ] = 0
        #    
        
        tmp     = numpy.copy(quarter)
        quarter = ndimage.binary_opening( quarter > quarter.mean(), structure=numpy.ones((5,5))).astype(float)
        quarter = ndimage.binary_dilation( quarter, structure=numpy.ones((3,3))).astype(float)
        
        if tmp[quarter > 0].sum() > 0:
            avg     = numpy.average( tmp[ quarter > 0] ) 
            quarter = quarter * ( tmp > avg )
            quarter = ndimage.binary_dilation( quarter > 0 , structure=numpy.ones((3,3))).astype(float)
        
        
        #
        u_ = numpy.arange(128*128)
        c  = u_ % 128
        r  = u_ // 128

        ''' 
        u = 1
        if side == "TL":
            cx = numpy.argmin( quarter[u,:] )
            cy = numpy.argmin( quarter[:,u] )
        elif side == "TR":
            cx = numpy.argmax( quarter[u,:] )
            cy = numpy.argmin( quarter[:,128-u] )
        elif side == "BL":
            cx = numpy.argmin( quarter[128-u,:] )
            cy = numpy.argmax( quarter[:,u] )
        else:
            cx = numpy.argmax( quarter[128-u,:] )
            cy = numpy.argmax( quarter[:,128-u] )
        '''
        
        cx, cy = getxy(quarter)
   
        '''
        #euclidean distance to cx, cy
        distance = numpy.sqrt( (c-cx)**2 + (r-cy)**2 )
        quarter_ = quarter.flatten()
        distance[ quarter_ < 1 ] = 128*numpy.sqrt(2)

        ind = numpy.argmin(distance)
        cx = ind % 128
        cy = ind // 128
        '''
        
        if numpy.linalg.norm(quarter) > 0:
            chull = convex_hull_image( quarter )                  
            edges = canny( chull, sigma = 4 )
        else:
            edges = numpy.zeros(quarter.shape)
            
        tol = 5

        #xy =  numpy.array( [ [j,i] for i in range(edges.shape[0]) for j in range(edges.shape[1]) if edges[i,j]>0 and abs(i-cy) < tol ] )
        xy =  numpy.array( [ [j,i] for i in range(max(cy-tol,0),min(cy+tol,128)) for j in range(edges.shape[1]) if edges[i,j]>0 ]) #and (i<128 and i>-1)] )

        #print('1',xy)
    
        xy_is_empty = xy.size == 0
        if xy_is_empty:
            side_y = cy
        else:
            side_y = ( (xy[:,1].min() + xy[:,1].max() )//2 ).astype(int)

        #xy = numpy.array( [ [j,i] for i in range(edges.shape[0]) for j in range(edges.shape[1]) if edges[i,j]>0 and abs(j-cx) < tol ] )
        xy = numpy.array( [ [j,i] for i in range(edges.shape[0]) for j in range(max(cx-tol,0), min(cx+tol,128)) if edges[i,j]>0 ]) #and (j<128 and j>-1) ] )

        #print('2',xy)
    
        
        xy_is_empty = xy.size == 0
        if xy_is_empty:
            topup_x = cx
        else:
            topup_x = ( (xy[:,0].min() + xy[:,0].max() )//2 ).astype(int)
            
        
        if side == "TL":
            corners = [ side_y, 0, 0, topup_x ]
        if side == "TR":
            corners = [ side_y, 127, 0, topup_x ]
            
        if side == "BL":
            corners = [ side_y, 0, 127, topup_x ]
        if side == "BR":
            corners = [ side_y, 127, 127, topup_x ]
            
        return cx, cy, corners


    CTIA_modules = [ CHIPS_TO_IGNORE_ANNOTATION[k][0] for k in range(len( CHIPS_TO_IGNORE_ANNOTATION))]
    CTIA_stripes = [ CHIPS_TO_IGNORE_ANNOTATION[k][1] for k in range(len( CHIPS_TO_IGNORE_ANNOTATION))]
    CTIA_chips   = [ CHIPS_TO_IGNORE_ANNOTATION[k][2] for k in range(len( CHIPS_TO_IGNORE_ANNOTATION))]
    CTIA_type    = [ CHIPS_TO_IGNORE_ANNOTATION[k][3] for k in range(len( CHIPS_TO_IGNORE_ANNOTATION))]
         
    for m in range(4):
        for s in range(6):
            imgstripe = get_image_stripe_detector( img, m, s)
            k         = s + m * 6
              
            for c in range(6):
                #print(m,s,c)
                
                chip = imgstripe[:,c*256:(c+1)*256]
                TL = chip[0:128,0:128]
                TR = chip[0:128,128:256]
                BL = chip[128:256,0:128]
                BR = chip[128:256,128:256]
                
                #### top-left corner                   
                x0, y0, cornersTL = _getxy_( TL, "TL" )                
                
                ys_TL, xs_TL, yb_TL, xb_TL = cornersTL
                
                #### top-right corner
                x1, y1, cornersTR = _getxy_( TR, "TR" )

                ys_TR, xs_TR, yb_TR, xb_TR = cornersTR
                
                #### bottom-left corner
                x2 , y2, cornersBL = _getxy_( BL, "BL" )

                ys_BL, xs_BL, yb_BL, xb_BL = cornersBL
                
                #### bottom-right corner
                x3, y3, cornersBR = _getxy_( BR, "BR")

                ys_BR, xs_BR, yb_BR, xb_BR = cornersBR
                #
                    
                x[0][k][2*c]   = x0 + c*256
                x[0][k][2*c+1] = 128+x1 + c*256
                y[0][k][2*c]   = y0
                y[0][k][2*c+1] = y1                
                
                x[1][k][2*c]   = x2 + c*256
                x[1][k][2*c+1] = 128+x3 + c*256
                y[1][k][2*c]   = 128 + y2
                y[1][k][2*c+1] = 128 + y3

                #
                xb[0][k][2*c]   = xb_TL + c*256
                xb[0][k][2*c+1] = 128 + xb_TR + c*256
                yb[0][k][2*c]   = yb_TL 
                yb[0][k][2*c+1] = yb_TR 

                xb[1][k][2*c]   = xb_BL + c*256
                xb[1][k][2*c+1] = 128 + xb_BR + c*256
                yb[1][k][2*c]   = 128 + yb_BL
                yb[1][k][2*c+1] = 128 + yb_BR

                #
                
                xs[0][k][2*c]   = xs_TL + c*256
                xs[0][k][2*c+1] = 128 + xs_TR + c*256
                ys[0][k][2*c]   = ys_TL 
                ys[0][k][2*c+1] = ys_TR 

                xs[1][k][2*c]   = xs_BL + c*256
                xs[1][k][2*c+1] = 128 + xs_BR + c*256
                ys[1][k][2*c]   = 128 + ys_BL
                ys[1][k][2*c+1] = 128 + ys_BR

                
                #if m==0 and s==1 and c==5:
                #if m==1 and s==4 and c==3:
                #if m==2 and s==4 and c==1:
                if False:   
                    row1, col1, row2, col2 = cornersTL
                    TL[y0, x0] = 1000
                    TL[row1, col1] = 1000
                    TL[row2, col2] = 1000
                    plt.imshow( TL) 
                    plt.show()

                    row1, col1, row2, col2 = cornersTR
                    TR[y1, x1] = 1000
                    TR[row1, col1] = 1000
                    TR[row2, col2] = 1000
                    plt.imshow( TR )
                    plt.show()

                    row1, col1, row2, col2 = cornersBL
                    BL[y2, x2] = 1000
                    BL[row1, col1] = 1000
                    BL[row2, col2] = 1000
                    plt.imshow( BL )
                    plt.show()

                    row1, col1, row2, col2 = cornersBR
                    BR[y3, x3] = 1000
                    BR[row1, col1] = 1000
                    BR[row2, col2] = 1000
                    plt.imshow( BR )
                    plt.show()
                    
                    sys.exit()

    # ignore boundary conditions

    for u in range( len( CTIA_modules ) ):

        s = CTIA_stripes[u]
        m = CTIA_modules[u]
        c = CTIA_chips[u]
        k = s + m * 6
        
        if CTIA_type[u] == "t":
            x[0][k][2*c]   = -1
            x[0][k][2*c+1] = -1
            y[0][k][2*c]   = -1
            y[0][k][2*c+1] = -1 
        else: #CTIA_type[u] == "b":
            x[1][k][2*c]   = -1
            x[1][k][2*c+1] = -1
            y[1][k][2*c]   = -1
            y[1][k][2*c+1] = -1
        
    #

    return x,y, xb, yb, xs, ys


def _worker_tracking540D_from_LNLS_template_modules( img, module, CTIA ):
    
    annotation = _worker_LNLS_template_with_squares_( img, CTIA )
    
    LNLS_SIZE_SQUARE = 6050
    LNLS_DIST_SQUARE = 8164

    x = annotation[0]   #x[0] top, x[1] bottom
    y = annotation[1]
    xb = annotation[2]
    yb = annotation[3]
    xs = annotation[4]
    ys = annotation[5]
        
    u = [ HORIZONTAL, VERTICAL, HORIZONTAL, VERTICAL ]
    
    
    #tracking = numpy.zeros((5 * 6 * 3, 6 + 8)) #5 stripe/interfaces x 3 annotations x 6 chips 

    stripes = [0,1,2,3,4] #stripes for interface comparison

    tracking          = numpy.zeros([1, 6 + 8])
    tracking_granular = numpy.zeros([1, 6 + 8])
    tracking_distance = numpy.zeros([1, 6 + 8])
    
    for j in stripes: #stripe or chip

        k = j + 6 * module
        
        #condition: boundary / horizontal
        j_ = (j + 1) % 6
        k_ = j_ + 6 * module
        
        for c in range(6): #chips

            if x[0][k][2*c] != -1  and x[1][k_][2*c] != -1:
                    
                ix1 = x[0][k][2*c]
                iy1 = 255 - y[0][k][2*c]
                iy1_= 1536 - (iy1 + j * 256) % 1536
                
                ix2 = x[1][k_][2*c] 
                iy2 = 255 - y[1][k_][2*c]
                iy2_= 1536 - ( iy2 + j_ * 256 )
                
                annot = numpy.array( [ ix1, iy1_,  module, j,
                                       iy1, ix2, iy2_,  module, j_, iy2, u[module], -1, k, k_] )
                tracking = numpy.vstack((tracking, annot))

                # granular
                
                ix1 = xb[0][k][2*c]
                iy1 = 255 - yb[0][k][2*c]
                iy1_= 1536 - (iy1 + j * 256) % 1536
                
                ix2 = xb[1][k_][2*c] 
                iy2 = 255 - yb[1][k_][2*c]
                iy2_= 1536 - ( iy2 + j_ * 256 )

                annot = numpy.array( [ ix1, iy1_,  module, j,
                                       iy1, ix2, iy2_,  module, j_, iy2, u[module], -1, k, k_] )
                tracking_granular = numpy.vstack((tracking_granular, annot))

            if x[0][k][2*c+1] != -1 and x[1][k_][2*c+1] != -1:

                ix1 = x[0][k][2*c+1]
                iy1 = 255 - y[0][k][2*c+1]
                iy1_= 1536 - (iy1 + j * 256) % 1536
                
                ix2 = x[1][k_][2*c+1] 
                iy2 = 255 - y[1][k_][2*c+1]
                iy2_= 1536 - ( iy2 + j_ * 256 )
                
                annot = numpy.array( [ ix1, iy1_,  module, j,
                                       iy1, ix2, iy2_,  module, j_, iy2, u[module], -1, k, k_] )
                tracking = numpy.vstack((tracking, annot))
                
                annot = numpy.array( [ ix1, iy1_,  module, j, iy1,
                                       ix2, iy2_,  module, j_, iy2, EUCLIDEAN, LNLS_SIZE_SQUARE, k, k_] )
                tracking_distance = numpy.vstack((tracking_distance, annot))

                #granular
                
                ix1 = xb[0][k][2*c+1]
                iy1 = 255 - yb[0][k][2*c+1]
                iy1_= 1536 - (iy1 + j * 256) % 1536
                
                ix2 = xb[1][k_][2*c+1] 
                iy2 = 255 - yb[1][k_][2*c+1]
                iy2_= 1536 - ( iy2 + j_ * 256 )
                
                annot = numpy.array( [ ix1, iy1_,  module, j,
                                       iy1, ix2, iy2_,  module, j_, iy2, u[module], -1, k, k_] )
                tracking_granular = numpy.vstack((tracking_granular, annot))
                
    #
    # tracking = [ix_P1, iy_P1, module_P1, stripe_P1, ystripe_P1, ix_P2, iy_P2, module_P2, stripe_P2, ystripe_P2, distance_type, distance ]
    # 
    return tracking[ ~ numpy.all(tracking == 0, axis=1) ],  tracking_granular[ ~ numpy.all(tracking_granular == 0, axis=1) ], tracking_distance[ ~ numpy.all(tracking_distance == 0, axis=1) ]



def tracking540D_from_LNLS_template( img , *args):

    if len(args)>0 and args[0] == "modules":
        #work tailored on another function ... :-(
        #this one is already complicated.
        module = args[1]
        if len(args)==2:
            CTIA = []
        else:
            CTIA   = args[2]

        return _worker_tracking540D_from_LNLS_template_modules( img, module, CTIA )
    
    annotation = _worker_LNLS_template_with_squares_( img )
    
    LNLS_SIZE_SQUARE = 6050
    LNLS_DIST_SQUARE = 8164

    x  = annotation[0]
    y  = annotation[1]
    xb = annotation[2]
    yb = annotation[3]
    xs = annotation[4]
    ys = annotation[5]

    u = [ VERTICAL, HORIZONTAL, VERTICAL, HORIZONTAL ]

    ug = [ HORIZONTAL, VERTICAL, HORIZONTAL, VERTICAL ]
    
    tracking          = numpy.zeros((1, 6 + 8))
    tracking_granular = numpy.zeros((1, 6 + 8))
    tracking_distance = numpy.zeros((1, 6 + 8))
    
    if not args:
        stripes = range(6)
        modules = range(4)
        task = "boundaries"
    else:
        task = args[0]
        if task == "boundaries":
            if len(args) == 1:
                modules = range(4)
                stripes = [1,2,3,4,5]
            else:
                bndry = args[1]
                
                if bndry == '0-1' or bndry == '1-0':
                    modules = [0]
                    stripes = [1,2,3,4,5]
                elif bndry == '1-2' or bndry == '2-1':
                    modules = [1]
                    stripes = [1,2,3,4,5]
                elif bndry == '2-3' or bndry == '3-2':
                    modules = [2]
                    stripes = [1,2,3,4,5]
                elif bndry == '3-0' or bndry == '0-3':
                    modules = [3]
                    stripes = [1,2,3,4,5]
                else:
                    modules = range(4)
                    stripes = [1,2,3,4,5]
                    
        elif task == "center": 
            modules = range(4)
            stripes = [0]
        else:
            modules = range(4)
            stripes = range(6)

    
    for m in modules: #range(4): #module
            
        for j in stripes: #stripe or chip

            k = j + 6 * m
            
            #condition: boundary / horizontal
            j_ = 0
            k_ = j_ + 6 * ((m+1) % 4)
            
            ix1 = x[1][k][11]
            iy1 = 255 - y[1][k][11]
            iy1_= 1536 - (iy1 + j * 256) % 1536

            ix2 = x[1][k_][ 11 - 2 * j] 
            iy2 = 255 - y[1][k_][ 11 - 2 * j]
            iy2_= 1536 - ( iy2 + j_ * 256 )

            #annotation: point (a) - point (b) should match at X/Y axis
            annot = numpy.array( [ ix1, iy1_,  m, j, iy1,
                                   ix2, iy2_, (m+1)%4, j_, iy2, u[m], -1, k, k_] )

            tracking = numpy.vstack((tracking, annot))

            #annotation: point (a) - point (b) have a known distance between them 
            annot = numpy.array( [ ix1, iy1_,  m, j, iy1,
                                   ix2, iy2_,  (m+1)%4, j_, iy2, EUCLIDEAN, LNLS_SIZE_SQUARE, k, k_] )
            
            tracking_distance = numpy.vstack((tracking_distance, annot))

            #
            #granular annotation
            if task == "center":
                j_ = 0
                k_ = j_ + 6 * ((m+3) % 4)
                
                ix1 = xb[1][k][11]
                iy1 = 255 - yb[1][k][11]
                iy1_= 1535 - (iy1 + j * 256) % 1536
                
                ix2 = xs[1][k_][ 11 ] #- 2 * j] 
                iy2 = 255 - ys[1][k_][ 11 ] # - 2 * j]
                iy2_= 1535 - ( iy2 + j_ * 256 )
                
                annot = numpy.array( [ ix1, iy1_,  m, j, iy1,
                                       ix2, iy2_, (m+3)%4, j_, iy2, ug[m], -1, k, k_] )
                
                tracking_granular = numpy.vstack((tracking_granular, annot))
            
            if task != "center":
                
                #condition: boundary / vertical 
                j_ = 0
                k_ = j_ + 6 * ((m+1) % 4)
                
                ix1 = x[0][k][11]
                iy1 = 255 - y[0][k][11]
                iy1_= 1536 - ( iy1 + j * 256 ) % 1536
                
                ix2 = x[1][k_][ 11 - (2 * j + 1)] 
                iy2 = 255 - y[1][k_][ 11 - (2 * j + 1)]             
                iy2_= ( iy2 + j_ * 256 )

                #annotation: point (c) - point (d) should match at X/Y axis
                annot = numpy.array( [ ix1, iy1_, m, j, iy1,
                                       ix2, iy2_, (m+1)%4, j_, iy2, u[m], -1, k, k_ ] )

                tracking = numpy.vstack((tracking, annot))

                #annotation: point (c) - point (d) have a known distance between them 
                annot  = numpy.array( [ ix1, iy1_, m, j, iy1,
                                        ix2, iy2_, (m+1)%4, j_, iy2, EUCLIDEAN, LNLS_SIZE_SQUARE, k, k_ ] )
                
                tracking_distance = numpy.vstack((tracking_distance, annot))
                
                #condition: distance between squares
                j_ = (j - 1) % 6
                k_ = (j_ + 6 * m) 
                
                ix1 = x[1][k][11]
                iy1 = 255 - y[1][k][11]
                iy1_= 1536 - (iy1 + j * 256) % 1536
                
                ix2 = x[0][k_][11] 
                iy2 = 255 - y[0][k_][11]
                iy2_= 1536 - ( iy2 + j_ * 256 ) % 1536

                #annotation: point (e) [bottom corner stripe j] - point (f) [top corner stripe j-1]  have a known distance between them
                annot = numpy.array( [ ix1, iy1_,  m, j,  iy1,
                                       ix2, iy2_,  m, j_, iy2, EUCLIDEAN, LNLS_SIZE_SQUARE, k, k_] )
                
                tracking_distance = numpy.vstack((tracking_distance, annot))
                
                #
                #
                # granular
                j_ = 0
                k_ = j_ + 6 * ((m+1) % 4)
                
                ix1 = xs[1][k][11]
                iy1 = 255 - ys[1][k][11]
                iy1_= 1536 - ( iy1 + j * 256 ) % 1536
                
                ix2 = xb[1][k_][ 11 - (2 * j)] 
                iy2 = 255 - yb[1][k_][ 11 - (2 * j)]             
                iy2_= ( iy2 + j_ * 256 )
                
                #annotation: point (c) - point (d) should match at X/Y axis
                annot = numpy.array( [ ix1, iy1_, m, j, iy1,
                                       ix2, iy2_, (m+1)%4, j_, iy2, u[m], -1, k, k_ ] )
                
                tracking_granular = numpy.vstack((tracking_granular, annot))

                
    #
    # tracking = [ix_P1, iy_P1, module_P1, stripe_P1, ystripe_P1, ix_P2, iy_P2, module_P2, stripe_P2, ystripe_P2, distance_type, distance ]
    # 
    return tracking[ ~ numpy.all(tracking == 0, axis=1) ], tracking_granular[ ~ numpy.all(tracking_granular == 0, axis=1) ], tracking_distance[ ~ numpy.all(tracking_distance == 0, axis=1) ]


def _constraint_rings_( x, * args):

    startEvalC = time.time()
    
    params = args[0]
    
    L0       = params[0]
    var      = params[1]
    x0       = params[2]
    tracking = params[3]
    boxinfo  = params[4]
    D        = params[5]
    eps      = params[6]
    radius   = params[7] 
    center   = params[8]
    pxl      = params[9]
    ell      = params[10]
    
    #y = ( normal0, center[0], center[1] )
    y          = x0
    y[var > 0] = x 
    
    det = get_detector_dictionary( y, L0 )
    
    data = tracking540D_vec_standard(det, tracking, (boxinfo, D) )

    #ell = EllipseModel()
    ell.estimate( data )
    xc, yc, a, b, theta = ell.params

    rad = (radius) * pxl

    tolerance = 1e-3
    
    #eccentricty = 0
    #if a > b:
    #    cons = numpy.sqrt( (a/rad) **2 - (b/rad)**2 )
    #else:
    #    cons = numpy.sqrt( (b/rad) **2 - (a/rad)**2 ) 
    
    
    #equality constraint
    cons = numpy.array( [numpy.abs(a - rad), numpy.abs(b - rad)] )
    
    print('oo> constraint function:', cons)
    print('oo> elapsed time for constraint function: {}'.format(time.time() - startEvalC ))

    return cons
    

def _criteria_rings( x, *args ):
 
    startEvalC = time.time()
    
    params = args[0]
    
    L0       = params[0]
    var      = params[1]
    x0       = params[2]
    tracking = params[3]
    boxinfo  = params[4]
    D        = params[5]
    eps      = params[6]
    radius   = params[7] 
    center   = params[8]
    pxl      = params[9]
    ell      = params[10]
    
    #y = ( normal0, center[0], center[1] )
    y          = x0
    y[var > 0] = x
    
    det = get_detector_dictionary( y, L0 )

    start = time.time()
    data = tracking540D_vec_standard(det, tracking, (boxinfo, D) )
    print('--> tracking:',time.time()-start)
    
    start=time.time()
    #ell = EllipseModel()
    ell.estimate( data )
    print('--> ell:',time.time()-start)
    
    xc, yc, a, b, theta = ell.params

    if False:
        t = numpy.linspace(0,2*numpy.pi,200)
        xt = xc + a*numpy.cos(theta)*numpy.cos(t) - b*numpy.sin(theta)*numpy.sin(t)
        yt = yc + a*numpy.sin(theta)*numpy.cos(t) + b*numpy.cos(theta)*numpy.sin(t)
        plt.plot(data[:,0], data[:,1],'o')
        plt.plot(xt, yt,'.-')
        plt.pause(0.25)
        #plt.show()
        #sys.exit()
    
    rad = (radius) * pxl

    
    if a > b:
        crit = ( (a) **2 - (b)**2 ) #+ ( (rad/a - 1)**2 + (rad/b - 1)**2 ) 
    else:
        crit = ( (b) **2 - (a)**2 ) #+ ( (rad/a - 1)**2 + (rad/b - 1)**2 )
    
    
    #crit = (a/rad - 1)**2 + (b/rad - 1)**2 

    #crit = (a - rad)**2 + (b - rad)**2

    #crit = (a*b/rad - 1)**2

    #crit = (a - b)**2 
    
    
    print('--> objective function:', crit)
    print('--> elapsed time for objective function: {}'.format(time.time() - startEvalC ))
    #print(a, b, rad)
    #print(x)
    
    #sys.exit()
    
    return crit


def _worker_optimize_tilting_540D_(x0, L0, variables, annotation, eps, radius , center, pxlsize):

    #plt.ion()

    pxl_tolerance = 100
    
    #-------------
    #optimization variables: ovar
    #
    ovar = optimization_variables(variables) 
    x0_  = x0[ovar > 0]    
    bnds = get_project_bounds_geometry( ovar )
    
    #print(bnds)
    #bnds[3] = ( center[0] - 100, center[0] + 100 ) 
    #bnds[4] = ( center[1] - 100, center[1] + 100 )

    for k in range(len(bnds)):
        bnds[k] = tuple( bnds[k]) 
    bnds = tuple(bnds)
     
    #print(bnds)
    #sys.exit()
    #print(annotation.astype(int))
    #sys.exit()

    ellipse = EllipseModel()
    
    #
    #boxinfo computed with project values (remain constat for optimization procedures)
    boxinfo = bbox540D( get_detector_dictionary( x0, L0  ) )
    J       = 256
    P       = 6
    M       = 6
    D       = parameters(J, P, M, False)
    #
    
    params = (L0, ovar, x0, annotation, boxinfo, D, eps, radius, center, pxlsize, ellipse ) 

    cons=({'type': 'eq', 'fun': lambda x, params: _constraint_rings_(x, params), 'args': (params,) } )
    
    #res = minimize( _criteria_rings, x0_, args=(params,),  bounds = bnds, constraints=cons, options={'maxiter':4000}, method="SLSQP", tol=1e-8)

    res = minimize( _criteria_rings, x0_, args=(params,),  bounds=bnds,  constraints=cons,  method="SLSQP", tol=1e-10 )
    
     
    #print('Iteration metadata: ', res)
    
    x = x0
    x[ ovar > 0] = res.x

    #
    det = get_detector_dictionary( x, L0 )
     
    data = tracking540D_vec_standard(det, annotation, (boxinfo, D) )
        
    ellipse.estimate( data )

    #xc, yc, a, b, theta = ellipse.params
    
    return x, res.x, ellipse.params



def _criteria_find_tilt_( x, *args ):

    startEvalC = time.time()
    
    params = args[0]
    
    data   = params[0]
    ell    = params[1]
    radius = params[2]
    L0     = params[3]
    
    rx = x[0]
    ry = x[1]
    rz = x[2]
    z =  x[3]
    
    n0 = numpy.sin(ry)*numpy.cos(rx)*numpy.cos(rz) + numpy.sin(rx)*numpy.sin(rz)
    n1 = numpy.sin(ry)*numpy.cos(rx)*numpy.sin(rz) - numpy.sin(rx)*numpy.cos(rz)
    n2 = numpy.cos(ry)*numpy.cos(rx)

    xp = data[:,0]
    yp = data[:,1]

    tilted      = numpy.zeros(data.shape)
    tilted[:,0] = (1-n0**2) * xp * z - n0 * n1 * yp * z - L0 * n0
    tilted[:,1] = - n0 * n1 * xp * z + (1 - n1**2) * yp * z - L0 * n1
    
    ell.estimate( tilted )

    xc, yc, a, b, theta = ell.params

    #plt.plot( tilted[:,0], tilted[:,1], '.' )
    #plt.xlim( [ xp.min(), xp.max() ])
    #plt.ylim( [ yp.min(), yp.min() + xp.max()-xp.min() ])
    #plt.pause(0.25)
    #plt.show()
    
    
    #crit = (a - radius)**2 + (b - radius)**2
    
    if a > b:
        crit = numpy.sqrt( (a/radius) **2 - (b/radius)**2 ) #+ ( (radius - a)**2 + (radius - b)**2 ) 
    else:
        crit = numpy.sqrt( (b/radius) **2 - (a/radius)**2 ) #+ ( (radius - a)**2 + (radius - b)**2 )
    
    
    #print('--> objective function:', crit)
    #print('--> elapsed time for objective function: {}'.format(time.time() - startEvalC ))
    print(a,b,radius)
    #print(x)
    return crit
    

    '''
    startEvalC = time.time()
    
    params = args[0]
    
    a   = params[0]
    b   = params[1]
    rad = params[2]

    rx = x[0]
    ry = x[1]
    rz = x[2]
    
    n0 = numpy.sin(ry)*numpy.cos(rx)*numpy.cos(rz) + numpy.sin(rx)*numpy.sin(rz)
    n1 = numpy.sin(ry)*numpy.cos(rx)*numpy.sin(rz) - numpy.sin(rx)*numpy.cos(rz)
    n2 = numpy.cos(ry)*numpy.cos(rx)

    f = numpy.zeros([3,])

    f[0] = ( (((1 - n1**2))**2)/(a**2) + ((n1*n0)**2)/(b**2) - 1.0/rad**2 ) / (n2**4)
    f[1] = ( n0*n1*((1 - n1**2))/(a**2) + n0*n1*((1 - n0**2))/(b**2)  ) / (n2**4)
    f[2] = ( (((1 - n0**2))**2)/(b**2) + ((n1*n0)**2)/(a**2) - 1.0/rad**2 )/ (n2**4)    
    crit = (( f )**2).sum() 
    
    print('--> objective function:', crit)
    print('--> elapsed time for objective function: {}'.format(time.time() - startEvalC ))
    
    return crit
    '''

def _constraint_rings_tilt_( x, * args):

    startEvalC = time.time()
    
    params = args[0]

    data   = params[0]
    ell    = params[1]
    radius = params[2]
    L0     = params[3]
    
    rx = x[0]
    ry = x[1]
    rz = x[2]
    z =  x[3]
    
    n0 = numpy.sin(ry)*numpy.cos(rx)*numpy.cos(rz) + numpy.sin(rx)*numpy.sin(rz)
    n1 = numpy.sin(ry)*numpy.cos(rx)*numpy.sin(rz) - numpy.sin(rx)*numpy.cos(rz)
    n2 = numpy.cos(ry)*numpy.cos(rx)

    xp = data[:,0]
    yp = data[:,1]

    tilted      = numpy.zeros(data.shape)
    tilted[:,0] = (1-n0**2) *  xp * z - n0 * n1 * yp * z - L0 * n0
    tilted[:,1] = - n0 * n1 *  xp * z + (1 - n1**2) *  yp * z - L0 * n1
    
    ell.estimate( tilted )

    #plt.plot( tilted[:,0], tilted[:,1], '.' )
    #plt.xlim( [ xp.min(), xp.max() ])
    #plt.xlim( [ yp.min(), yp.max() ])
    #plt.pause(0.25)
    #plt.show()
    
    xc, yc, a, b, theta = ell.params
    
    #equality constraint
    cons = numpy.array( [numpy.abs(a - radius), numpy.abs(b - radius)] )
    
    #print('oo> constraint function:', cons)
    #print('oo> elapsed time for constraint function: {}'.format(time.time() - startEvalC ))

    return cons
    

def _worker_find_tilt_540D_( x, L0, pointsAtDevice, radius, pxlsize ):

    #plt.ion()
    
    #
    #boxinfo computed with project values (remain constat for optimization procedures)
    det     = get_detector_dictionary( x, L0 )
    boxinfo = bbox540D( det )
    J       = 256
    P       = 6
    M       = 6
    D       = parameters(J, P, M, False)
    
    data = tracking540D_vec_standard(det, pointsAtDevice, (boxinfo, D) )

    ellipse = EllipseModel()
        
    params = (data, ellipse, radius * pxlsize, L0) 
    x0     = [0,0,0,1]

    bnds = ( (-5,5), (-5,5) , (-5,5), (0.5,1.5) )

    cons=({'type': 'eq', 'fun': lambda x, params: _constraint_rings_tilt_(x, params), 'args': (params,) } )
        
    res = minimize( _criteria_find_tilt_, x0, args=(params,),  method="SLSQP", constraints=cons )

    #
    rx = res.x[0]
    ry = res.x[1]
    rz = res.x[2]
    z  = res.x[3]
    
    n0 = numpy.sin(ry)*numpy.cos(rx)*numpy.cos(rz) + numpy.sin(rx)*numpy.sin(rz)
    n1 = numpy.sin(ry)*numpy.cos(rx)*numpy.sin(rz) - numpy.sin(rx)*numpy.cos(rz)
    n2 = numpy.cos(ry)*numpy.cos(rx)

    xp          = data[:,0]
    yp          = data[:,1]
    tilted      = numpy.zeros(data.shape)
    tilted[:,0] = (1-n0**2) *  xp * z - n0 * n1 * yp * z - L0  * n0
    tilted[:,1] = - n0 * n1 * xp * z + (1 - n1**2) * yp * z - L0 * n1
    
    ellipse.estimate( tilted )
    
    xc, yc, a, b, theta = ellipse.params

    cx = (1-n0**2) * xc * z - n0 * n1 * yc * z  - L0 * n0
    cy = - n0 * n1 * xc * z + (1 - n1**2) * yc * z - L0 * n1

    print(cx,cy)
    
    cx = int( cx/pxlsize ) 
    cy = int( cy/pxlsize ) 
    
    return numpy.array([rx,ry,0]), z, [cx, cy], [a, b, theta] 

    

def _worker_extract_rings_from_restoration( x, back, geometry, n , cristal, energy, distance, center ):

    if cristal == "AgBe":
        # Silver Behenate
        # http://gisaxs.com/index.php/Material:Silver_behenate        
        # (peak references)
        
        q_from_wiki = 0.1077
        distDet     = distance * 1000      #microns
        q           = q_from_wiki * 1e+10  # m^-1
        energy      = energy * 1000        # ev
        cvel        = 299792458            # m/s
        planck      = 4.135667662e-15      # ev * s
        wavelength  = cvel * planck/ (energy) 
        theta       = numpy.arcsin( q * wavelength / (4*numpy.pi) )  

        distFirstRing = numpy.tan( 2 * theta) * distDet
        pxlSize       = geometry['pxlsize']
        distFirstRing = int( distFirstRing / pxlSize )
        
        radius = n * distFirstRing
        eps    = 0.35 * distFirstRing

        CENTER = [1536 + center[0], 1536 + center[1] ]
        x = numpy.array(range(back.shape[1]))
        y = numpy.flipud( numpy.array(range(back.shape[0]))) 
        xx,yy = numpy.meshgrid(x,y)
        ring = ((numpy.sqrt((xx-CENTER[0])**2 + (yy-CENTER[1])**2)<radius+eps) &
                (numpy.sqrt((xx-CENTER[0])**2 + (yy-CENTER[1])**2)>radius-eps)).astype(numpy.double)

        back[back < 0] = 0
        
        plt.imshow( back * ring ) # + 90 *  ring )
        plt.show()
        #sys.exit()
        
        #ring from restoration
        #rfrest = ndimage.gaussian_filter( back * ring, sigma=0.5, order=0 )
        
        imgs = [ (back*ring)[0:1536,0:1536], (back*ring)[0:1536,1536:3072],
                 (back*ring)[1536:3072,1536:3072], (back*ring)[1536:3072,0:1536]  ]  

        rings = [ (ring)[0:1536,0:1536], (ring)[0:1536,1536:3072],
                  (ring)[1536:3072,1536:3072], (ring)[1536:3072,0:1536]]

        for m in range(4):
            tmp = imgs[m][rings[m]>0].ravel()
            
            hist, bin_edges = numpy.histogram( tmp , bins=int(tmp.max()), density=False)

            #removing background value
            hist[0] = 0
            
            maxvalue = numpy.argmax(hist)
            #print(maxvalue)

            #plt.figure(0)
            #plt.imshow( imgs[m] )
            #plt.figure(1)
            #plt.plot( hist )
            #plt.show()

            extract =  ( imgs[m] >  maxvalue ).astype(numpy.double)
            extract = ndimage.gaussian_filter( extract > (1-0.95) * extract.max() , sigma=0.5, order=0 )
            
            imgs[m] = extract
            #plt.imshow( extract )
            #plt.show()

        extract = numpy.vstack(( numpy.hstack((imgs[0],imgs[1])), numpy.hstack((imgs[3],imgs[2])) ))

        plt.imshow( extract )
        plt.show()
        #sys.exit()
        
        #cleaning the signal
        
        extract = ndimage.binary_closing(  extract > extract.mean(), structure=numpy.ones((2,2)) ) #.astype(numpy.double)
        extract = ndimage.binary_opening( extract, structure=numpy.ones((3,3))).astype(float)
        extract = ( extract == extract.max() ).astype(numpy.double)

        extract = ndimage.gaussian_filter(extract, sigma=0.95, order=0)
        extract = ( extract > 0 ).astype(numpy.double)
        
        plt.imshow(extract)
        plt.show()
        #sys.exit()
        
        #approximate radius from extract
        eps = max ( ( numpy.diag(extract) > 0 ).sum() / 2 ,  ( numpy.diag(numpy.flipud(extract)) > 0 ).sum() / 2 )  
 
        #print(eps, radius)
        #sys.exit()
        
        return extract, eps, radius
    
def _worker_extract_rings_tracking_points( forward ) :

    img_for_annotation = _worker_annotation_image( forward )

    #plt.imshow(img_for_annotation)
    #plt.show()
    #sys.exit()
    
    xy = numpy.array( [ [j,i] for i  in range(3072) for j in range(3072) if img_for_annotation[i,j] > 0 ]  )
    #plt.plot(xy[:,0], xy[:,1], 'o')
    #plt.show()
    
    reasonable_no_points = 400 
    N = xy.shape[0]
    #print(N)
    if N > 2 * reasonable_no_points :
        # remove points: it is not necessary a huge number of points for fitting purposes!
        step = N//reasonable_no_points
        xy = xy[0:N:step, :]    
    
    #print(xy.shape)
    #plt.plot(xy[:,0], xy[:,1], 'o')
    #plt.show()
    #sys.exit()
    
    tracking = annotation_points_standard ( xy )

    def reject_outliers(data, m=6):
        #considering x-xis
        data = data[ abs(data[:,0] - numpy.mean(data[:,0])) < m * numpy.std(data[:,0])  ]
        #considering y-xis
        data = data[ abs(data[:,1] - numpy.mean(data[:,1])) < m * numpy.std(data[:,1])  ]
        return data
        
    tracking = reject_outliers( tracking, m=2)

    #plt.plot(tracking[:,0], tracking[:,1], 'o')
    #plt.show()
    #sys.exit()
    
    return tracking



def optimize_geometry( large, short, *args ):

    if not args:
        input_annotation = False
    else:
        input_annotation = True
        extra            = args[0]
        
    #------------------------------------
    #Reminder:
    #(C)hips (t)o (i)gnore (a)nnotation:
    #pointwise [ [module, stripe, chip, <t> or <b> ] , ... ]
    #------------------------------------ 

    CTIA_l     = large['ctia']
    img_l      = numpy.clip( large['measure'], 0, large['threshold'] )
    distance_l = large['distance']
   
    CTIA_s     = short['ctia']
    img_s      = numpy.clip( short['measure'], 0, short['threshold'] ) 
    distance_s = short['distance']

    load = True

    if load == False:
        start = time.time()

        print('--> Automatic annotation for 2 x LNLS template images ...')
        
        tracking_l_c, granular_l_c, trackdist_l_c = tracking540D_from_LNLS_template ( img_l , 'center' )    
        tracking_l_b, granular_l_b, trackdist_l_b = tracking540D_from_LNLS_template ( img_l, 'boundaries' )
        tracking_l_0, granular_l_0, trackdist_l_0 = tracking540D_from_LNLS_template ( img_l, 'modules', 0, CTIA_l )
        tracking_l_1, granular_l_1, trackdist_l_1 = tracking540D_from_LNLS_template ( img_l, 'modules', 1, CTIA_l )
        tracking_l_2, granular_l_2, trackdist_l_2 = tracking540D_from_LNLS_template ( img_l, 'modules', 2, CTIA_l )
        tracking_l_3, granular_l_3, trackdist_l_3 = tracking540D_from_LNLS_template ( img_l, 'modules', 3, CTIA_l )
        
        tracking_s_c, granular_s_c, trackdist_s_c = tracking540D_from_LNLS_template ( img_s , 'center' )
        tracking_s_b, granular_s_b, trackdist_s_b = tracking540D_from_LNLS_template ( img_s, 'boundaries' )
        tracking_s_0, granular_s_0, trackdist_s_0 = tracking540D_from_LNLS_template ( img_s, 'modules', 0, CTIA_s )
        tracking_s_1, granular_s_1, trackdist_s_1 = tracking540D_from_LNLS_template ( img_s, 'modules', 1, CTIA_s)
        tracking_s_2, granular_s_2, trackdist_s_2 = tracking540D_from_LNLS_template ( img_s, 'modules', 2, CTIA_s )
        tracking_s_3, granular_s_3, trackdist_s_3 = tracking540D_from_LNLS_template ( img_s, 'modules', 3, CTIA_s )
        
        tracking = {
            'large': {
                'c': [tracking_l_c, granular_l_c, trackdist_l_c],
                'b': [tracking_l_b, granular_l_b, trackdist_l_b],
                '0': [tracking_l_0, granular_l_0, trackdist_l_0],
                '1': [tracking_l_1, granular_l_1, trackdist_l_1],
                '2': [tracking_l_2, granular_l_2, trackdist_l_2],
                '3': [tracking_l_3, granular_l_3, trackdist_l_3],
            },
            'short': {
                'c': [tracking_s_c, granular_s_c, trackdist_s_c],
                'b': [tracking_s_b, granular_s_b, trackdist_s_b],
                '0': [tracking_s_0, granular_s_0, trackdist_s_0],
                '1': [tracking_s_1, granular_s_1, trackdist_s_1],
                '2': [tracking_s_2, granular_s_2, trackdist_s_2],
                '3': [tracking_s_3, granular_s_3, trackdist_s_3],
            }
        }

        elapsed = time.time() - start
        
        print(' ... done! {} min'.format(elapsed/60.))

        with open('tracking.pickle', 'wb') as f:
            pickle.dump(tracking, f)
                            
        sys.exit()
    else:
        
        with open('tracking.pickle', 'rb') as f:
            tracking = pickle.load(f)

        tracking_l_c  = tracking['large']['c'][0]
        granular_l_c  = tracking['large']['c'][1]
        trackdist_l_c = tracking['large']['c'][2]
        tracking_l_b  = tracking['large']['b'][0]
        granular_l_b  = tracking['large']['b'][1]
        trackdist_l_b = tracking['large']['b'][2]        
        tracking_l_0  = tracking['large']['0'][0]
        granular_l_0  = tracking['large']['0'][1]
        trackdist_l_0 = tracking['large']['0'][2]
        tracking_l_1  = tracking['large']['1'][0]
        granular_l_1  = tracking['large']['1'][1]
        trackdist_l_1 = tracking['large']['1'][2]
        tracking_l_2  = tracking['large']['2'][0]
        granular_l_2  = tracking['large']['2'][1]
        trackdist_l_2 = tracking['large']['2'][2]
        tracking_l_3  = tracking['large']['3'][0]
        granular_l_3  = tracking['large']['3'][1]
        trackdist_l_3 = tracking['large']['3'][2]
        
        tracking_s_c  = tracking['short']['c'][0]
        granular_s_c  = tracking['short']['c'][1]
        trackdist_s_c = tracking['short']['c'][2]
        tracking_s_b  = tracking['short']['b'][0]
        granular_s_b  = tracking['short']['b'][1]
        trackdist_s_b = tracking['short']['b'][2]
        tracking_s_0  = tracking['short']['0'][0]
        granular_s_0  = tracking['short']['0'][1]
        trackdist_s_0 = tracking['short']['0'][2]
        tracking_s_1  = tracking['short']['1'][0]
        granular_s_1  = tracking['short']['1'][1]
        trackdist_s_1 = tracking['short']['1'][2]
        tracking_s_2  = tracking['short']['2'][0]
        granular_s_2  = tracking['short']['2'][1]
        trackdist_s_2 = tracking['short']['2'][2]
        tracking_s_3  = tracking['short']['3'][0]
        granular_s_3  = tracking['short']['3'][1]
        trackdist_s_3 = tracking['short']['3'][2]

    #

    #print('normal')
    #print(tracking_l_c.astype(int))
    #print('dist')
    #print(trackdist_l_c.astype(int))
    #print('granular')
    #print(granular_l_3.astype(int))
    
    tracking_granular_l   = numpy.vstack((granular_l_c, granular_l_b, granular_l_0, granular_l_1, granular_l_2, granular_l_3))

    tracking_l = numpy.vstack((trackdist_l_c, trackdist_l_b, trackdist_l_0, trackdist_l_1, trackdist_l_2, trackdist_l_3,
                               tracking_l_c, tracking_l_b, tracking_l_0, tracking_l_1, tracking_l_2, tracking_l_3 ))

    trackdist_l = numpy.vstack((trackdist_l_c, trackdist_l_b, trackdist_l_0, trackdist_l_1, trackdist_l_2, trackdist_l_3,
                                granular_l_c, granular_l_b, granular_l_0, granular_l_1, granular_l_2, granular_l_3 ))
    
    tracking_s = numpy.vstack((trackdist_s_c, trackdist_s_b, trackdist_s_0, trackdist_s_1, trackdist_s_2, trackdist_s_3,
                               tracking_s_c, tracking_s_b, tracking_s_0, tracking_s_1, tracking_s_2, tracking_s_3))

    tracking_granular_s = numpy.vstack((granular_s_c, granular_s_b, granular_s_0, granular_s_1, granular_s_2, granular_s_3))
    

    ''' 
    ##see annotated points
    disp     = tracking_l
    img      = img_l
    dist     = distance_l
    x0       = get_project_values_geometry('project')
    track    = annotation_image_points( disp )
    #plt.imshow( img_l + track * 500)
    #plt.show()
    annotation_ux_restoration(img + 500 * track, disp, dist, 1536, x0)
    sys.exit()
    #
    #
    '''
    
    x = set_optimization_variable( get_project_values_geometry('project') )
    
    variables = {
        'a': 4*[0],
        'rx': 24*[0], 
        'ry': 24*[0],
        'rz': 24*[0],
        'offset': 24*[0],
        'ox': 24*[1],  
        'oy': 24*[1],
        'normal': 3*[0],
        'center': 2*[0],
        'z': 0
    }

    x = _worker_optimize_geometry_540D_1dist( x , variables, tracking_l, distance_l )
 
    print('--> Fitting at long distance')
    print('--> ox:', x['ox'])
    print('--> oy:', x['oy'])

    annotation_ux_restoration(img_l , tracking_l, distance_l, 1536, get_optimization_variable( x ) )
              
    tolerance = {'ox': [(2)*MEDIPIX, (2)*MEDIPIX], 'oy': [(2)*MEDIPIX, (2)*MEDIPIX] ,
                 'rx': [0.5, 0.5] , 'ry': [5/1000., 5/1000.], 'rz': [0.1, 0.1] }
    
    var_l = {
        'a': 4*[0],
        'rx': 24*[1], 
        'ry': 24*[1],
        'rz': 24*[1],
        'offset': 24*[0],
        'ox': 24*[1],   
        'oy': 24*[1],
        'normal': 3*[0],
        'center': 2*[0],
        'z': 0
    }
    
    x = _worker_optimize_geometry_540D_1dist( x , var_l, trackdist_l, distance_l, tolerance)
    
    print('--> Fitting at long distance')
    print('--> rx:', x['rx'])
    print('--> ry:', x['ry'])
    print('--> rz:', x['rz'])
    print('--> ox:', x['ox'])
    print('--> oy:', x['oy'])
    
    annotation_ux_restoration(img_l , tracking_granular_l, distance_l, 1536, get_optimization_variable( x ) )

    annotation_ux_restoration(img_s , tracking_s, distance_s, 1536, get_optimization_variable( x ) )
    
    sys.exit()
    
    #ANGLES

    '''
    x = set_optimization_variable( get_project_values_geometry('project') )
    x['ox'] = [205.86666671, 234.19929424, 230.61359106, 190.06388522, 199.8262727,
               223.59649503, 152.49413884, 189.82181656, 190.99168855, 159.54318735,
               123.96292709, 142.82486717, 264.65722539, 155.54900107, 298.94580806,
               263.03181203, 319.3004465,  225.28608995, 196.06045161, 187.99610669,
               239.57376855, 212.76689554, 140.73680125, 182.74192213]
    
    x['oy'] = [ 578.94010541, 772.19953675,  854.84756214, 1102.35682257, 1404.19988675,
                1540.3769321,  413.35229965,  771.99217147,  799.29756268, 1139.91260565,
                1200.62329679, 1595.30465623,  592.29993094,  895.81762205,  923.64336736,
                1280.94305346, 1308.02778451, 1444.41623425,  524.69611957,  717.5640991,
                854.78817484, 1267.18264634, 1293.90067334, 1650. ]
    
    #x['rx'] = [-6.75000002, -6.75000003, -6.75,       -6.74999999, -6.77237008, -6.74999996,
    #           -5.75000001, -6.74999999, -6.75000002, -5.79599783, -6.33680127, -6.74999999,
    #           -6.75000001, -6.75,       -6.75000004, -6.74999999, -6.75,       -5.79462346,
    #           -5.75000001, -5.897136,   -6.75,       -6.75,       -6.6602369,  -6.68953176 ]

    #x['ry'] = [-0.07773921, -0.06563388, -0.07694307, -0.06678297, -0.06161121, -0.04281501,
    #           -0.09245844, -0.09069175, -0.07258544, -0.06644541, -0.05963691, -0.04783509,
    #           0.07185643,  0.07846862,  0.08159435,  0.07604335,  0.06423851,  0.03210082,
    #           0.06110764, 0.06989252,  0.08742639,  0.07169574,  0.10204919,  0.07306205]
    
    #x['rz'] = [-3.76441923e-09, -1.28018703e-01,  6.48263763e-02,  2.53757344e-09,
    #           -1.93575441e-02, -9.43972474e-02, -1.14896202e-01,  1.86313141e-08,
    #           -1.00759642e-01,  3.13095899e-02, -2.85041939e-09, -3.45194290e-09,
    #           1.81180372e-01,  5.04705799e-02, -6.48459675e-02, -7.29598316e-02,
    #           8.69795624e-10,  1.99999996e-01,  4.89466010e-09,  1.49530105e-01,
    #           -7.91925552e-02,  9.04846861e-02, -1.45804010e-01, -9.57010884e-10] 
    '''
    #print(x)
    #annotation_ux_restoration(img_s , tracking_s, distance_s, 1536, get_optimization_variable( x ) )
    #print(x)
    #annotation_ux_restoration(img_l , tracking_l, distance_l, 1536, get_optimization_variable( x ) )

    var_s = {
        'a': 4*[0],
        'rx': 24*[1], 
        'ry': 24*[0],
        'rz': 24*[1],
        'offset': 24*[1],
        'ox': 24*[0], 
        'oy':  24*[0],
        'normal': 3*[0],
        'center': 2*[1],
        'z': 0
    }
    
    tolerance = {'rx': [1, 1] , 'ry': [5/1000., 5/1000.], 'rz': [0.1, 0.1] }
    
    y = _worker_optimize_geometry_540D_1dist( x , var_s, tracking_s, distance_s, tolerance)

    print('--> Fitting at short distance')
    print('--> rx:', y['rx'])
    print('--> ry:', y['ry'])
    print('--> rz:', y['rz'])
    print('--> ox:', y['ox'])
    print('--> oy:', y['oy'])
    print('--> normal:', y['normal'])
    print('--> offset:', y['offset'])
    print('--> center:', y['center'])

    annotation_ux_restoration(img_s , tracking_s, distance_s, 1536, get_optimization_variable( y ) )
    
    y = _worker_optimize_geometry_540D_1dist( y , var_s, tracking_granular_s, distance_s, tolerance)

    print('--> Fitting at short distance')
    print('--> rx:', y['rx'])
    print('--> ry:', y['ry'])
    print('--> rz:', y['rz'])
    print('--> ox:', y['ox'])
    print('--> oy:', y['oy'])
    print('--> normal:', y['normal'])
    print('--> offset:', y['offset'])
    print('--> center:', y['center'])

    x['rx'] = y['rx']
    x['ry'] = y['ry']
    x['rz'] = y['rz']
    
    print(get_optimization_variable( y ))
    print(get_optimization_variable( x ))
    
    annotation_ux_restoration(img_s , tracking_granular_s, distance_s, 1536, get_optimization_variable( y ) )
    
    annotation_ux_restoration(img_l , tracking_granular_l, distance_l, 1536, get_optimization_variable( x ))

    ###
    
    '''
    if input_annotation:

        no_annotation = len(extra)

        variables = {
            'a': 4*[0],
            'rx': 24*[1], 
            'ry': 24*[0],
            'rz': 24*[0],
            's': 8 *[0],
            'offset': 4*[0],
            'ox': 24*[0], 
            'oy':  24*[0],
            'normal': 3*[0],
            'center': 2*[0],
            'z': 0
        }   
        
        for k in range(no_annotation):
        
            tracking_user, _ = annotation_points( extra[k]['annotation'] )
            distance_user    = extra[k]['distance']
            
            start = time.time()
            
            x, out = _worker_optimize_geometry_540D_1dist( x , variables, tracking_user, distance_user)
            
            elapsed = time.time() - start 
            print('--> Fitting user input: {} sec'.format(elapsed) )


    annotation_ux_restoration(img_l , tracking_l, distance_l, 1536, x)
    '''
    
    return x


def optimize_geometry_sstar( dic, *args ):

    '''
    aimg = pi540D._worker_annotation_image ( img )
    aimg = ndimage.gaussian_filter( aimg, sigma=0.95, order=0 )
    aimg = aimg / aimg.max()
    
    #edges = canny(aimg / aimg.max() , sigma= 4)       
    #aimg= skeletonize (edges , method='lee')
    
    imgs = [ (aimg)[0:1536,0:1536],       (aimg)[0:1536,1536:3072],
        (aimg)[1536:3072,1536:3072], (aimg)[1536:3072,0:1536]  ]  
    
    eps = 20

    signal = imgs[1][1536 - eps,:]
    where  = 1.0 * (signal > 0.98 )
    where  = numpy.where( where == where.max() )[0]

    k_ = 0
    where_ = []
    for k in range(len(where)-1):
    if where[k+1] - where[k] > 10:
        idx = int( numpy.mean(where[k_:k+1]) )
        where_.append( idx  )
        k_ = k+1
        imgs[1][ 1536 - eps , idx ] =  -100


    eps = 10
        
    signal = imgs[2][eps,:]
    where  = 1.0 * (signal > 0.98 )
    where  = numpy.where( where == where.max() )[0]
    
    k_ = 0
    where_ = []
    for k in range(len(where)-1):
    if where[k+1] - where[k] > 10:
        idx = int( numpy.mean(where[k_:k+1]) )
        where_.append( idx  )
        k_ = k+1
        imgs[2][ eps , idx ] =  -100

    plt.imshow(imgs[2])
    plt.show()
    sys.exit()
    '''


    '''
    new = numpy.copy(img)
    plot = True
    
    edges = canny(img / img.max() , sigma= 4)       
    sk = skeletonize (edges , method='lee')
           
    def reject_outliers(data, m=6):
    data = data[ abs( data - numpy.mean(data) ) < m * numpy.std(data)  ]
    return data

    for k in range(24):
    stripe = pi540D.get_stripe_from_measure_540D(new, k )
    
    for j in range(6):
        chip = stripe[:,j * 256 : (j+1)*256]
        #maxv = chip.max()
        #minv = chip.min()
        #eps = 0.1 * (maxv - minv)
        #chip = chip - eps

        image = numpy.copy(chip) / chip.max()

        waste = 10
        image[0:waste,:] = 0
        image[:,0:waste] = 0
        image[256-waste:256,:] = 0
        image[:,256-waste:256] = 0

        edges = canny(image, sigma= 4)        
        #edges = ndimage.gaussian_filter( edges, sigma=0.1, order=0 )
        skeleton = skeletonize (edges, method='lee')
        
        #image = edges * image
        
        #dges = ndimage.binary_fill_holes(edges).astype(int)
        
        #edges = ndimage.binary_dilation( edges, structure=numpy.ones((3,3))).astype(float)
        
        
        #image = ndimage.binary_opening( image, structure=numpy.ones((3,3))).astype(float)
        
        extract = numpy.zeros(chip.shape)
        
        # Set a precision of 0.5 degree.
        tested_angles = numpy.linspace(-numpy.pi / 2, numpy.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(image, theta=tested_angles)
        
        allAngles = []
        distance = []
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            allAngles.append( angle )
            distance.append( dist )
        
        goodAngles = reject_outliers( numpy.array(allAngles), m=12 ) # len(allAngles) )
        
        x = range(256)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            (x0, y0) = dist * numpy.array([numpy.cos(angle), numpy.sin(angle)])
            #print(x0,y0, dist)
            if angle in goodAngles:
                a = numpy.tan( angle + numpy.pi/2 )
                y = ( a * x + ( y0 - a * x0 ) + numpy.random.random(1)[0] * 3 ).astype(int)
                y[ y >= 256] = 255 
                y[ y < 0 ] = 0
                for q in range(256):
                    extract[y[q], x[q]] = 1
                    
        #plt.imshow( image + extract  ) 
        #plt.imshow(numpy.hstack((image, extract)))
        #plt.show()


        #plt.imshow(numpy.hstack(( image, edges, skeleton)))
        #plt.show()
        
        
        stripe[:, j*256:(j+1)*256] =  skeleton

    if False:
        if k==2:
            sys.exit()
        
    pi540D.set_stripe_from_measure_540D(new, stripe, k)
    

    plt.imshow( numpy.hstack (( new , img, sk )) )
    plt.show()
    sys.exit()
    '''

    dbeam = dic['directbeam']
    img   = dic['measure']
    flat  = dic['flat']
    bkg   = dic['background']
    L     = dic['distance']
    
    img = ( img - bkg ) * flat
    img[ img < 0] = -1
    img[ numpy.isnan(img) ] = -1
    
    img = numpy.clip( img , 0, dic['threshold'] )

    dbeam = numpy.clip( dbeam, 0, 10 )

    aimg = _worker_annotation_image ( dbeam )

    aimg = ndimage.gaussian_filter( aimg, sigma=0.95, order=0 )
    aimg = aimg/aimg.max()
    
    aimg = 1.0 * ( aimg > 0.98 )

    u = numpy.array(range(3072))
    xx,yy = numpy.meshgrid(u,u)
    
    xc = (aimg * xx).sum() / aimg.sum()
    yc = (aimg * yy).sum() / aimg.sum()
    
    plt.imshow(aimg)
    plt.show()

    print(xc, yc)
    
    #edges = canny( aimg , sigma= 4)        
    #skeleton = skeletonize (edges, method='lee')
    #plt.imshow( skeleton )
    #plt.show()
    #sys.exit()
    
    xc_annotated = 2094
    yc_annotated = 2142

    xdet    = get_project_values_geometry()
    project = get_detector_dictionary( xdet, L )
    
    project['center'] = [ xc_annotated - 1536 , - ( yc_annotated - 1536) ]
    
    start = time.time()
    geometry = geometry540D( project )
    elapsed_geo = round( time.time() - start, 3)
    
    annotation = numpy.array([ [2094, 2140] ])
    tracking = annotation_points_standard ( annotation )
    tracking = tracking540D_vec_standard ( project, tracking ) 
    
    x0 = int( tracking[0][2] )
    y0 = int( tracking[0][3] ) 
    print(x0, y0)
    #back[ y0, x0 ] = 0

    ## -------
    
    u = numpy.linspace(-1,1,3072)
    xx,yy = numpy.meshgrid(u,u)
    nlines = 24 * 2
    th0 = 4.5 * numpy.pi / 180.
    th = numpy.linspace(th0,numpy.pi+th0,nlines,endpoint=False)
    
    z = numpy.zeros([3072, 3072])
    for k in range(nlines):
        width = (2.0/3072) * 10
        line = numpy.abs( (xx-u[x0]) * numpy.cos(th[k]) + (yy - u[y0] )* numpy.sin(th[k])  ) < width
        z[ line ] = 1

    z[ ( (xx-u[x0])**2 + (yy - u[y0])**2 < 0.2**2 ) ] = 0
    z = z/z.max()

    forw = forward540D ( z, geometry)

    plt.imshow( forw * img ) 
    plt.show()
    
    ###
    '''  
    start = time.time()
    
    x = get_project_values_geometry('project')
    
    variables = {
        'a': 4*[0],
        'rx': 24*[1], 
        'ry': 24*[0],
        'rz': 24*[0],
        's': 8 *[1],
        'L': 4*[0],
        'ox': 24*[0], 
        'oy': 24*[0],
        'normal': 3*[0],
        'center': 2*[0],
        'z': 0
    }
    
    start = time.time()
    
    x, out = _worker_optimize_geometry_540D_1dist( x , variables, tracking_l, distance_l)
    
    elapsed = time.time() - start
    print('--> Fitting angles and shift at long distance: {} sec'.format(elapsed) )
    #print('-->', out)

    annotation_ux_restoration(img_l , tracking_l, distance_l, 1536, x)
    '''
    
    return x



def optimize_tilt( rings, flat, mask ):
            
    #### find module global tilt

    if rings['type'] != 'AgBe':
        print('Error: go find a measure from AgBe using X-rays!')
    else:        
        x = get_project_values_geometry()
        L = rings['distance']
        e = rings['energy']
        img = numpy.clip( rings['measure'], 0, rings['threshold'] )
        center = rings['center']

        #print(mask.shape)
        #print(flat.shape)
        #print(rings['measure'].shape)
        #img_ = rings['measure'][mask > 0.5] / flat[ mask > 0.5]
        #img_ = rings['measure'] * flat 
        #plt.imshow(numpy.clip(img_, 0, rings['threshold']) )
        #plt.show()
        #sys.exit()

        
        project  = get_detector_dictionary(x, L )
        geometry = geometry540D( project )
        back     = backward540D ( img, geometry ) 
        pxlsize  = geometry['pxlsize']

        #plt.imshow(back)
        #plt.show()
        
        for n in [6]: #[2,3,4,5]:

            #x[136:139] = [1.2, 0 ,0]
            #xc = 0
            #yc = 0
            #x[141] = 1 # 0.98
            #dL = 20
            
            extract, eps, radius = _worker_extract_rings_from_restoration(x, back, geometry, n , rings['type'], e, L, center )        
            
            forward = correct_image_forward_540D(extract, geometry)
        
            pointsAtDevice = _worker_extract_rings_tracking_points( forward ) 

            #print(pointsAtDevice.shape)
            #sys.exit()
            
            '''
            stripesTouched = list( set( pointsAtDevice[:,3] ) )
            modulesTouched = list( set( pointsAtDevice[:,2] ) ) 
            rx = 24*[0]
            for m in range(len(modulesTouched)):
                for j in range(len(stripesTouched)):
                    stripeNumber = set_module_strip (m, j)
                    rx[stripeNumber] = 1

            print(rx)
            #sys.exit()
            '''
            
            normal, z, [cx,cy], [a,b,theta] = _worker_find_tilt_540D_( x, L, pointsAtDevice, radius, pxlsize )
            print('-->', normal, z, cx, cy)
            print('-->', a, b, theta )

            #cx = cx - 2
            
            #normal = numpy.array([ -0.02402593, -0.08225017,  0.        ])
            #z = 0.9789
            #cx = 4.61
            #cy = -3.75

            #
            project['normal'] = normal
            project['z'] = 1.0/z
            project['center'] = [cx, cy]
            geometry = geometry540D( project )
            new_back = backward540D ( img, geometry )
            new_backe = backward540D ( forward, geometry ) 
            
            xbox = geometry['boxinfo']['xbox']
            u = numpy.linspace(xbox[0],xbox[1],3072)
            ybox = geometry['boxinfo']['ybox']
            v = numpy.linspace(ybox[0],ybox[1],3072)
            dxy = pxlsize
            xx,yy = numpy.meshgrid(u,v)

            rx = normal[0]
            ry = normal[1]
            rz = normal[0]
            
            n0 = numpy.sin(ry)*numpy.cos(rx)*numpy.cos(rz) + numpy.sin(rx)*numpy.sin(rz)
            n1 = numpy.sin(ry)*numpy.cos(rx)*numpy.sin(rz) - numpy.sin(rx)*numpy.cos(rz)
            n2 = numpy.cos(ry)*numpy.cos(rx)
            
            xx_ = (1-n0**2) * xx * (1/z) - n0 * n1 *  yy * (1/z) - L  * n0
            yy_ = - n0 * n1 * xx * (1/z) + (1 - n1**2) * yy * (1/z) - L * n1

            ix = ( (xx_ - xbox[0]) / dxy ).astype(int)
            iy = ( (yy_ - ybox[0]) / dxy ).astype(int)

            ix[  ix < 0 ] = 0
            ix[  ix >= 3072 ] = 3072 - 1
            iy[  iy < 0 ] = 0
            iy[  iy >= 3072 ] = 3072 - 1
    
            new = back[iy, ix]
                        
            u = numpy.linspace(0,3071,3072,endpoint=True)
            xx,yy = numpy.meshgrid(u,u)
            yy = numpy.flipud(yy)
            CENTER = [1536  + cx, 1536 + cy ]
            print('---> CENTER:', CENTER)
            ring = ((numpy.sqrt(((xx-CENTER[0]))**2 + ((yy-CENTER[1]))**2)<radius+eps) &
                    (numpy.sqrt(((xx-CENTER[0]))**2 + ((yy-CENTER[1]))**2)>radius-eps)).astype(numpy.double)
            
            plt.imshow( new + 90 * ring)
            #plt.imshow( new + 90 * ring)

            #plt.imshow( new + 90 * ring)
            #plt.imshow(ring)
            plt.show()
            
            '''
            variables = {
                'a': 4*[0],
                'rx': 24*[0], 
                'ry': 24*[0],
                'rz': 24*[0],
                's': 8 *[0],
                'L': 4*[1],
                'ox': 24*[0], 
                'oy':  24*[0],
                'normal': [1,1,1], ##[1, 1, 1],
                'center': 2*[0],
                'z': 1
            }            

            x, out, [xc, yc, a, b, theta] = _worker_optimize_tilting_540D_( x, L, variables, pointsAtDevice, eps, radius, center, pxlsize)
            
            #x[136] = 2.7 
            #x[137] = 0.2
            #
            project  = get_detector_dictionary(x, L)
            geometry = geometry540D( project )
            back     = backward540D ( img, geometry )
            backe     = backward540D ( forward, geometry ) 
            
            cx=xc/pxlsize
            cy=yc/pxlsize
            print('--->',cx,cy)
            u = numpy.linspace(0,3071,3072)
            xx,yy = numpy.meshgrid(u,u)
            CENTER = [1536  + cy, 1536 + cx ]
            ring = ((numpy.sqrt(((xx-CENTER[0]))**2 + ((yy-CENTER[1]))**2)<radius+eps) &
                    (numpy.sqrt(((xx-CENTER[0]))**2 + ((yy-CENTER[1]))**2)>radius-eps)).astype(numpy.double)
            
            plt.imshow(back + 90 * ring)
            #plt.imshow( back - backe * back )
            plt.show()        
            ''' 
    return x

