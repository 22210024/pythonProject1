import cv2

img = cv2.imread("E:/png/1.png")  # Read image

# Defining all the parameters
t_lower = 100  # Lower Threshold
t_upper = 200  # Upper threshold
aperture_size = 5  # Aperture size
L2Gradient = True  # Boolean

# Applying the Canny Edge filter
# with Aperture Size and L2Gradient
edge = cv2.Canny(img, t_lower, t_upper,
                 apertureSize=aperture_size,
                 L2gradient=L2Gradient)
cv2.namedWindow('original',0)
cv2.namedWindow('edge',0)
cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('E:/png/1gray.png',edge)