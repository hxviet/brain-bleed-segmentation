import nibabel as nib
import numpy as np


def background_segmentation_mask(input_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Create a segmentation mask where all voxels are set to class 0 (background).

    The segmentation mask maintains spatial alignment with the source image via the affine matrix.
    
    Args:
        input_img (nibabel.Nifti1Image): Source NIfTI-1 image
    
    Returns:
        nibabel.Nifti1Image: Background segmentation mask as NIfTI-1 image
    """
    
    # Create a background mask using shape from header
    shape = input_img.header.get_data_shape()
    background_mask = np.zeros(shape, dtype=np.uint8)
    
    # Create a new nifti file with proper header settings for segmentation
    mask_nifti = nib.Nifti1Image(background_mask, input_img.affine, input_img.header)
    
    # Copy relevant header fields from original image
    mask_header = mask_nifti.header
    
    # Set intent to "label", which is appropriate for segmentation masks
    mask_header.set_intent('label')
    
    # Remove any scaling factors that might be present in the original CT scan
    mask_header['scl_slope'] = 1.0
    mask_header['scl_inter'] = 0.0
    
    # Set datatype to uint8 for the mask
    mask_header.set_data_dtype(np.uint8)
    
    # Set sform and qform codes to 2 (aligned)
    mask_header.set_sform(None, code=2)
    mask_header.set_qform(None, code=2)
    
    return mask_nifti
