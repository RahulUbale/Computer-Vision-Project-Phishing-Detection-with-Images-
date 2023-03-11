import numpy as np
from PIL import Image
from scipy.fftpack import fft, dct
import sys
from matplotlib import pyplot as plt

# Color Layout Descriptor (CLD) designed to capture the spatial distribution of color in an image
# CLD is compact, so it it great for fast browsing and search algorithms
# CLD is resolution invariant

# calculate CLD vector (1,129)
def get_cld(img): 
    c = img.width
    r = img.height
    
    # partition image into 8x8
    r = np.ceil(r/8)
    c = np.ceil(c/8)
    
    r8 = int(8*r)
    c8 = int(8*c)
    #print(int(r8), int(c8))
    
    # resize image so it's divisible by 8
    # size parameter is (width, height)
    im = img.resize((c8,r8))
    
    # will be used in block processing
    #ones = np.ones((r,c))
    
    # define mask to identify DC coef of DCT
    # element (0,0) is DC coef
    mask = np.zeros((8,8))
    mask[0,0] = 1
    
    # find average of each block
    im_block_avg = blockproc(im)
    
    # convert image to 64x64
    im_avg_resized = im_block_avg.resize((64,64))
    
    # convert to YCbCr channel
    YCbCr = im_avg_resized.convert('YCbCr')
    #y, cb, cr = YCbCr.split() # to get individual components
    
    # apply Discrete Cosine Transform (DCT) to each block
    # return the DC coefficients of each channel from each block
    Y_dct, Cb_dct, Cr_dct = blockproc_dct(YCbCr)
    # cld is 1x192
    cld = Y_dct + Cb_dct + Cr_dct

    return cld, im_block_avg, im_avg_resized, YCbCr
    


def blockproc(img, blockshape=(8,8)):
    bwidth, bheight = blockshape
    # determine number of pixels in each block
    pix_w = int(img.width/bwidth)
    pix_h = int(img.height/bheight)
    # print(f'pix_w: {pix_w}\npix_h: {pix_h}')
    # create new image to store results
    new_im = Image.new(mode='RGB',size=(img.width, img.height), color=0)
    # iterate find the average of each pixel in the given block
    for bw in range(bwidth):
        for bh in range(bheight):
            # store each pixel value
            r_arr = []
            g_arr = []
            b_arr = []
            for p_w in range(pix_w):
                for p_h in range(pix_h):
                    curr_pix = (p_w+(pix_w*bw), p_h+(pix_h*bh))
                    r, g, b = img.getpixel(curr_pix)
                    #print(img.getpixel(curr_pix))
                    r_arr.append(r)
                    g_arr.append(g)
                    b_arr.append(b)
                    # new_im.putpixel(curr_pix, (r, g, b)) # test to reproduce original image
            # calculate mean for each channel
            r_mean = int(np.mean(r_arr))
            g_mean = int(np.mean(g_arr))
            b_mean = int(np.mean(b_arr))
            # update new image by setting all pixels in current block to mean values
            for p_w in range(pix_w):
                for p_h in range(pix_h):
                    curr_pix = (p_w+(pix_w*bw), p_h+(pix_h*bh))
                    new_im.putpixel(curr_pix, (r_mean, g_mean, b_mean))
    return new_im
            

def blockproc_dct(img, blockshape=(8,8)):
    bwidth, bheight = blockshape
    # determine number of pixels in each block
    pix_w = int(img.width/bwidth)
    pix_h = int(img.height/bheight)
    # store dct values; only return the first coef from each block
    Y_dct = []
    Cb_dct = []
    Cr_dct = []
    # iterate find the average of each pixel in the given block
    for bw in range(bwidth):
        for bh in range(bheight):
            # store each pixel value
            Y_arr = np.ones(blockshape)
            Cb_arr = np.ones(blockshape)
            Cr_arr = np.ones(blockshape)
            for p_w in range(pix_w):
                for p_h in range(pix_h):
                    curr_pix = (p_w+(pix_w*bw), p_h+(pix_h*bh))
                    Y, Cb, Cr = img.getpixel(curr_pix)
                    Y_arr[p_h,p_w] = Y
                    Cb_arr[p_h,p_w] = Cb
                    Cr_arr[p_h,p_w] = Cr
            # compute DCT of current block for each channel
            curr_Y_dct = dct(Y_arr)
            curr_Cb_dct = dct(Cb_arr)
            curr_Cr_dct = dct(Cr_arr)
            # store first coeff
            Y_dct.append(curr_Y_dct[0,0])
            Cb_dct.append(curr_Cb_dct[0,0])
            Cr_dct.append(curr_Cr_dct[0,0])
    return Y_dct, Cb_dct, Cr_dct
    

def cld_main(im_path):
    # load image
    #im1 = Image.open(sys.argv[1])
    #im2 = Image.open(sys.argv[2])
    
    im1 = Image.open(im_path)
    #im2 = Image.open('inputs/example_resized.jpg')
    #im2 = Image.open('inputs/example_noise.jpg')
    #im2 = Image.open('inputs/example_cropped.jpg')
    #im2 = Image.open('inputs/robin.jpg')
    #im2 = Image.open('phishIRIS_DL_Dataset/phishIRIS_DL_Dataset/train/adobe/adobe (6).png')
    
    if im1.mode != 'RGB':
        im1 = im1.convert('RGB')
    #print(im1.mode)
    
    # process image 1
    cld, im_avg, im_avg_64x64, YCbCr = get_cld(im1)
    #im_avg.save('outputs/imgavg.jpg')
    #im_avg_64x64.save('outputs/imgavg64x64.jpg')
    #YCbCr.save('outputs/YCbCr.jpg')
    
    # # process image 2
    # cld2, im_avg2, im_avg_64x64_2, YCbCr2 = get_cld(im2)
    # im_avg2.save('outputs/imgavg2.jpg')
    # im_avg_64x64_2.save('outputs/imgavg64x64_2.jpg')
    # YCbCr2.save('outputs/YCbCr_2.jpg')

    # plot the dct coefficients from images
    #plt.plot(cld)
    #plt.plot(cld2)
    #plt.savefig('outputs/plot_tiger_vs_airplane.png')
    

    return cld
    # im3 = im1.resize((300,250))
    # im3.save('inputs/example_resized.jpg')
    

    # # separate CLD components for weighted distance
    # cld_Y1 = cld1[0:64]
    # cld_Cb1 = cld1[64:128]
    # cld_Cr1 = cld1[128:192]
    
    # cld_Y2 = cld1[0:64]
    # cld_Cb2 = cld1[64:128]
    # cld_Cr2 = cld1[128:192]
    
    # # calculate weighted distance between both matrices
    # # weights are Y:2, Cb:2, Cr:4
    
    # weighted_dist = np.sqrt(np.sum(2*((cld_Y1-cld_Y2)**2))) + np.sqrt(np.sum(2*((cld_Cb1-cld_Cb2)**2))) + np.sqrt(np.sum(4*((cld_Cr1-cld_Cr2)**2)))
    
    # # L1 Distance
    # l1_dist = np.abs(cld1 - cld2)
    
    # # L2 Distance
    # l2_dist = np.linalg.norm(cld1 - cld2)

    # plot cld

if __name__ == '__main__':
    # Load an image
    cld = cld_main(im_path='adobe (1).png')


    
























