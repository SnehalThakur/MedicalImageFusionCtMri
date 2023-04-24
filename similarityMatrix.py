# pip install opencv-python
# pip install sewar

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

import cv2

# Mean Squared Error (MSE)
# Root Mean Squared Error (RMSE)
# Peak Signal-to-Noise Ratio (PSNR)
# Structural Similarity Index (SSIM)
# Universal Quality Image Index (UQI)
# Multi-scale Structural Similarity Index (MS-SSIM)
# Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)
# Spatial Correlation Coefficient (SCC)
# Relative Average Spectral Error (RASE)
# Spectral Angle Mapper (SAM)
# Visual Information Fidelity (VIF)
img1 = cv2.imread(r"C:\Users\snehal\PycharmProjects\MedicalImageFusion\multimodal-image-fusion-to-detect-brain-tumors\dataset\Patient Data\p1\fusion.jpg")
img2 = cv2.imread(r"C:\Users\snehal\PycharmProjects\MedicalImageFusion\multimodal-image-fusion-to-detect-brain-tumors\dataset\Patient Data\p1\mri_registered.jpg")

# Uncomment below 4 lines to see images
# cv2.imshow("image1", img1)
# cv2.imshow("image2", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("MSE: ", mse(img2,img1))
print("RMSE: ", rmse(img2, img1))
print("PSNR: ", psnr(img2, img1))
print("SSIM: ", ssim(img2, img1))
print("UQI: ", uqi(img2, img1))
print("MSSSIM: ", msssim(img2, img1))
print("ERGAS: ", ergas(img2, img1))
print("SCC: ", scc(img2, img1))
# print("RASE: ", rase(img2, img1))
print("SAM: ", sam(img2, img1))
print("VIF: ", vifp(img2, img1))