#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-Light Image Enhancement using Gamma Correction Prior in Mixed Color Spaces

This implementation follows the paper's approach:
1. Invert the dark image to transform it into a "hazy" image
2. Use dehazing techniques to remove the "haze"
3. Invert back to get the enhanced bright image

The key insight: A dark image, when inverted, looks like a bright scene covered in haze.
So we can use fog/haze removal methods to brighten dark images!

Created on Fri Jul 21 14:50:00 2022
@author: TripleJ
"""
import numpy as np
import cv2

from skimage import morphology
import time
import os
import argparse

# Command-line arguments setup - allows you to specify input/output folders and settings
parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_dir', dest='InputPath', default='test_images', help='directory for testing inputs')
parser.add_argument('--output_dir', dest='OutputPath', default='real_image_results', help='directory for testing outputs')
parser.add_argument('--gamma_max', dest='gamma_max', type=float, default=6.0, help='gamma_max value')
args = parser.parse_args()

#==============================================================================
# UTILITY FUNCTIONS - Convert between different number ranges
#==============================================================================

def i2f(i_image):
    """
    Convert pixel values from 0-255 range to 0-1 range (for easier math)
    Integer to Float: Scale pixel values from [0, 255] to [0, 1]
    """
    f_image = np.float32(i_image)/255.0
    return f_image

def f2i(f_image):
    """
    Convert pixel values from 0-1 range back to 0-255 range (for saving images)
    Float to Integer: Scale back from [0, 1] to [0, 255]
    """
    i_image = np.uint8(f_image*255.0)
    return i_image

#==============================================================================
# ATMOSPHERIC LIGHT ESTIMATION
# Purpose: Find the "brightest haze" in the inverted image (darkest area in original)
# This tells us how much darkness needs to be removed
#==============================================================================

def Compute_A_Tang(im):
    """
    Step 1 of paper: Estimate how much "darkness" to remove
    
    Plain English: Look at the darkest parts of the image and measure them.
    This tells us the overall darkness level that needs correction.
    
    How it works:
    - Find the darkest pixels in each local area (called "dark channel")
    - Select the top 1% darkest regions
    - Measure their average color - this is our "darkness to remove"
    """
    erosion_window = 15  # Size of local area to examine (15x15 pixels)
    n_bins = 200         # Number of brightness levels to analyze

    # Split image into Red, Green, Blue channels
    R = im[:, :, 2]
    G = im[:, :, 1]
    B = im[:, :, 0]

    # DARK CHANNEL: For each small region, find the darkest pixel
    # This reveals areas that are severely underexposed
    dark = morphology.erosion(np.min(im, 2), morphology.square(erosion_window))

    # Find the TOP 1% darkest regions in the image
    [h, edges] = np.histogram(dark, n_bins, [0, 1])
    numpixel = im.shape[0]*im.shape[1]
    thr_frac = numpixel*0.99  # We want the darkest 1% (99th percentile)
    csum = np.cumsum(h)
    nz_idx = np.nonzero(csum > thr_frac)[0][0]
    dc_thr = edges[nz_idx]
    mask = dark >= dc_thr
    
    # From these darkest regions, get their RGB color values
    rs = R[mask]
    gs = G[mask]
    bs = B[mask]

    # Calculate the MEDIAN (middle value) of these dark regions
    # This represents the "atmospheric light" - the overall darkness to subtract
    A = np.zeros((1,3))
    A[0, 2] = np.median(rs)  # Red channel
    A[0, 1] = np.median(gs)  # Green channel
    A[0, 0] = np.median(bs)  # Blue channel

    return A

#==============================================================================
# IMAGE ANALYSIS HELPERS
# Purpose: Measure different properties of the image (brightness, channels)
#==============================================================================

def GetIntensity(fi):
    """Calculate average brightness: Add R+G+B and divide by 3"""
    return cv2.divide(fi[:, :, 0] + fi[:, :, 1] + fi[:, :, 2], 3)

def GetMax(fi):
    """Get the brightest color channel at each pixel (max of R, G, or B)"""
    max_rgb = cv2.max(cv2.max(fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    return max_rgb

def GetMin(fi):
    """Get the darkest color channel at each pixel (min of R, G, or B)"""
    min_rgb = cv2.max(cv2.min(fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    return min_rgb

#==============================================================================
# GAMMA CORRECTION - The Heart of the Paper's Innovation
# Purpose: Brighten dark areas MORE than bright areas (adaptive enhancement)
#==============================================================================

def PixelAdaptiveGamma(InputImg, NormImg, amax):
    """
    PIXEL-ADAPTIVE GAMMA CORRECTION - Key innovation from the paper!
    
    Plain English: Make dark areas MUCH brighter, but keep bright areas similar.
    Different parts of the image get different amounts of brightening.
    
    How it works:
    - Very dark pixels get a HIGH gamma (strong brightening, up to amax=6)
    - Bright pixels get a LOW gamma (minimal change, close to 1)
    - This prevents over-brightening of already-bright areas
    
    Think of it like: Turning up the brightness knob MORE in shadows, LESS in highlights
    """
    GCImg = np.empty(InputImg.shape, InputImg.dtype)
    amin = 1      # Minimum gamma (no change for bright areas)
    xmax = 1      # Brightness threshold for "bright pixels"
    xmin = 0      # Brightness threshold for "dark pixels"
    
    # Find the brightest channel at each pixel
    Imax = GetMax(InputImg)

    # Calculate spatially-varying gamma using exponential function
    # This creates a smooth transition from high gamma (dark) to low gamma (bright)
    a = (amax - amin) / (np.exp(-xmin) - np.exp(-xmax))
    b = amax - a * np.exp(-xmin)
    g2 = a * np.exp(-Imax) + b
    g1 = np.where(Imax < xmin, amax, g2)
    gamma = np.where(Imax > xmax, amin, g1)

    # Apply the calculated gamma to each color channel
    # Formula: output = input^gamma
    # When gamma > 1: brightens the image
    # When gamma = 1: no change
    for ind in range(0, 3):
        GCImg[:, :, ind] = NormImg[:, :, ind] ** gamma

    return GCImg

#==============================================================================
# TRANSMISSION MAP ESTIMATION - Measure "how much darkness" at each pixel
# Purpose: Create a map showing which areas need more/less enhancement
#==============================================================================

def EstimateTransmission(InputImg, NormImg, gamma_max=6):
    """
    TRANSMISSION MAP - Shows "how dark" each pixel is
    
    Plain English: Create a map where:
    - Dark areas get LOW transmission values (need lots of brightening)
    - Bright areas get HIGH transmission values (need little brightening)
    
    This is the paper's main contribution - using gamma correction to estimate
    how much each pixel is affected by darkness (like haze affects fog photos)
    
    Steps:
    1. Measure image before gamma correction (original darkness)
    2. Measure image after gamma correction (enhanced version)
    3. Compare them to see HOW MUCH each area changed
    4. Areas that changed a lot = very dark = low transmission
    """
    T_min = 0.1  # Minimum transmission (don't make it completely black)
    me = np.finfo(np.float32).eps  # Tiny number to avoid division by zero
    
    # BEFORE gamma correction: measure brightness
    hi = GetIntensity(NormImg)      # Average brightness of each pixel
    hmax = GetMax(NormImg)          # Brightest channel at each pixel

    # Apply GAMMA CORRECTION
    GCImg = PixelAdaptiveGamma(InputImg, NormImg, gamma_max)

    # AFTER gamma correction: measure brightness again
    ji = GetIntensity(GCImg)        # New average brightness
    jmax = GetMax(GCImg)            # New brightest channel

    # CALCULATE TRANSMISSION using the paper's formula
    # This compares before/after to see how much each area was affected
    tn = np.maximum(jmax * hi - hmax * ji, me)
    td = np.maximum((jmax - ji) * hi, me)
    Tmap = 1.0 - hi * (tn / td)
    
    # Clip to valid range [T_min, 1.0]
    # T_min ensures we don't darken too much
    return np.clip(Tmap, T_min, 1.0)

#==============================================================================
# IMAGE RECOVERY - Apply the enhancement
# Purpose: Use the transmission map to actually brighten the image
#==============================================================================

def Recover(im, tmap, A):
    """
    SCENE RECOVERY - The actual brightening step!
    
    Plain English: Remove the darkness using our calculated transmission map.
    
    This uses the "atmospheric scattering model" from dehazing research:
    - Original formula: I = J*t + A*(1-t)  [How fog obscures images]
    - Solving for J: J = (I - A*(1-t)) / t  [Remove the fog]
    
    In our case:
    - I = inverted dark image (looks hazy)
    - J = recovered bright image (dehazed)
    - t = transmission map (how much darkness)
    - A = atmospheric light (overall darkness level)
    
    We're essentially "removing haze" to brighten the dark image!
    """
    res = np.empty(im.shape, im.dtype)
    
    # Apply the dehazing formula to each color channel (R, G, B)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - 1.0 + A[0, ind]) / tmap + 1.0 - A[0, ind]

    # Ensure all values stay in valid range [0, 1]
    return np.clip(res, 0.0, 1.0)

#==============================================================================
# POST-PROCESSING - Final touch-ups
# Purpose: Adjust contrast and brightness for natural-looking results
#==============================================================================

def Adjust(im, perh, perl):
    """
    HISTOGRAM STRETCHING - Final contrast adjustment
    
    Plain English: Make sure we're using the full brightness range.
    
    Sometimes after enhancement, all pixels bunch up in the middle range.
    This step spreads them out from darkest (0) to brightest (255).
    
    perh = 99.5 means: ignore the top 0.5% brightest pixels (could be noise)
    perl = 0.5 means: ignore the bottom 0.5% darkest pixels (could be noise)
    """
    aim = np.empty(im.shape, im.dtype)
    im_h = np.percentile(im, perh)  # Find the 99.5th percentile (almost brightest)
    im_l = np.percentile(im, perl)  # Find the 0.5th percentile (almost darkest)
    
    # Stretch: map [im_l, im_h] to [0, 1]
    for ind in range(0, 3):
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)

    return np.clip(aim, 0.0, 1.0)

def Normalize(im):
    """
    CHANNEL-WISE NORMALIZATION
    
    Plain English: Adjust each color (R, G, B) separately to balance them.
    
    This prevents color casts (unwanted tints) by ensuring each color channel
    uses its full range. Helps maintain natural-looking colors after enhancement.
    """
    aim = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im_h = np.max(im[:, :, ind])    # Brightest pixel in this channel
        im_l = np.min(im[:, :, ind])    # Darkest pixel in this channel
        # Stretch this channel to full [0, 1] range
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)
        aim[:, :, ind] = np.clip(aim[:, :, ind], 0.0, 1.0)

    return aim

#==============================================================================
# MAIN PIPELINE - Putting it all together!
# This follows the paper's algorithm step-by-step
#==============================================================================

def main(InputImg, gamma_max):
    """
    ==========================================================================
    MAIN ENHANCEMENT ALGORITHM - Complete Step-by-Step Process
    ==========================================================================
    
    Following the paper: "Low-light image enhancement using gamma correction
    prior in mixed color spaces"
    
    THE BIG IDEA:
    Dark images, when inverted, look like bright images covered in fog/haze.
    So we can use fog removal techniques to brighten dark images!
    
    PIPELINE OVERVIEW:
    1. Invert the dark image → looks like a hazy bright image
    2. Estimate "haze level" (atmospheric light) → how much darkness to remove
    3. Calculate transmission map → where the darkness is concentrated
    4. Remove the "haze" → brighten the image
    5. Invert back → get the final enhanced image
    6. Adjust contrast → make it look natural
    """
    start_time = time.time()
    
    #==========================================================================
    # STEP 1: PREPARATION
    #==========================================================================
    print("Step 1: Converting and inverting image...")
    
    # Convert from [0,255] to [0,1] for easier math
    float_image = i2f(InputImg)
    
    # INVERSION: Dark → Bright (Key transformation!)
    # A dark image becomes bright when inverted
    # Then apply slight blur to reduce noise
    DenoiseImg = 1.0 - cv2.GaussianBlur(float_image, (7, 7), 0)
    # Now DenoiseImg looks like a hazy/foggy bright image
    
    #==========================================================================
    # STEP 2: ESTIMATE ATMOSPHERIC LIGHT
    #==========================================================================
    print("Step 2: Estimating overall darkness level...")
    
    # Find the "haze level" - how much darkness affects the whole image
    A = Compute_A_Tang(DenoiseImg)
    print(f"  Atmospheric light (darkness level): R={A[0,2]:.3f}, G={A[0,1]:.3f}, B={A[0,0]:.3f}")
    
    # Normalize by atmospheric light - remove overall darkness bias
    NormImg = np.empty(float_image.shape, float_image.dtype)
    for ind in range(0, 3):
        NormImg[:, :, ind] = DenoiseImg[:, :, ind] / A[0, ind]
    NormImg = Normalize(NormImg)
    
    #==========================================================================
    # STEP 3: ESTIMATE TRANSMISSION MAP (The Paper's Key Innovation!)
    #==========================================================================
    print("Step 3: Calculating transmission map using gamma correction prior...")
    
    # This is where the paper's innovation happens!
    # Use gamma correction to figure out which areas are darker
    Transmap = EstimateTransmission(float_image, NormImg, gamma_max)
    print(f"  Transmission range: min={Transmap.min():.3f}, max={Transmap.max():.3f}")
    
    #==========================================================================
    # STEP 4: RECOVER SCENE (Remove the "Haze" = Brighten the Dark Image)
    #==========================================================================
    print("Step 4: Recovering enhanced image...")
    
    # Apply the transmission map to remove darkness
    RecoverImg = Recover(float_image, Transmap, A)
    
    #==========================================================================
    # STEP 5: POST-PROCESSING (Final Adjustments)
    #==========================================================================
    print("Step 5: Final contrast adjustment...")
    
    # Stretch histogram for better contrast and natural look
    AdjustImg = Adjust(RecoverImg, 99.5, 0.5)
    
    #==========================================================================
    # DONE!
    #==========================================================================
    end_time = time.time()
    print(f"✓ Enhancement complete in {end_time - start_time:.3f} seconds")
    print("="*70)
    
    return AdjustImg

#==============================================================================
# BATCH PROCESSING - Process multiple images from a folder
#==============================================================================

if __name__ == "__main__":
    """
    BATCH PROCESSING MODE
    
    When you run this script directly, it will:
    1. Read all images from the input directory
    2. Enhance each one using the algorithm
    3. Save results to the output directory
    
    Usage:
      python paper_implementation.py --input_dir test_images --output_dir results --gamma_max 6.0
    """
    
    print("\n" + "="*70)
    print("LOW-LIGHT IMAGE ENHANCEMENT - Paper Implementation")
    print("="*70)
    print(f"Input directory:  {args.InputPath}")
    print(f"Output directory: {args.OutputPath}")
    print(f"Gamma max value:  {args.gamma_max}")
    print("="*70 + "\n")
    
    # Create output directory if it doesn't exist
    OutputPath = args.OutputPath
    os.makedirs(OutputPath, exist_ok=True, mode=0o777)
    
    # Find all image files in the input directory
    # Supports: jpg, jpeg, png, bmp, tiff, webp
    FileList = [file for file in os.listdir(args.InputPath) if
                    (file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg"))
                    or file.endswith(".webp") or file.endswith(".tiff") or file.endswith(".tif")
                    or file.endswith(".bmp") or file.endswith(".png")]
    
    print(f"Found {len(FileList)} images to process\n")
    
    # Process each image
    for FileNum in range(0, len(FileList)):
        print(f"\n[{FileNum+1}/{len(FileList)}] Processing: {FileList[FileNum]}")
        print("-"*70)
        
        # Read input image
        FilePathName = args.InputPath + '/' + FileList[FileNum]
        InputImg = cv2.imread(FilePathName, cv2.IMREAD_COLOR)
        
        if InputImg is None:
            print(f"  ✗ Error: Could not read {FileList[FileNum]}")
            continue
        
        print(f"  Image size: {InputImg.shape[1]}x{InputImg.shape[0]} pixels")
        
        # Enhance the image using the paper's algorithm
        OutputImg = main(InputImg, args.gamma_max)
        
        # Save result
        Name = os.path.splitext(FileList[FileNum])
        output_path = OutputPath + '/' + Name[0] + '.png'
        cv2.imwrite(output_path, f2i(OutputImg))
        print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "="*70)
    print(f"✓ All {len(FileList)} images processed successfully!")
    print(f"✓ Results saved in: {OutputPath}")
    print("="*70 + "\n")
