import icp_2d
import cv2
import numpy as np

source_mask = np.zeros((500,500))
source_mask[200:300,200:300] = 1

cv2.imshow("source mask", source_mask)

# target_mask = np.zeros((500,500))
# target_mask[300:500,300:500] = 1

size = 500  # Define the size of the mask
target_mask = np.zeros((size, size))

# # Define the center and size of the diamond
# center = (size // 2, size // 2)
# half_diagonal = 100  # Half of the diagonal length of the diamond

# # Define diamond vertices
# pts = np.array([
#     (center[0], center[1] - half_diagonal),  # Top
#     (center[0] - half_diagonal, center[1]),  # Left
#     (center[0], center[1] + half_diagonal),  # Bottom
#     (center[0] + half_diagonal, center[1])   # Right
# ], np.int32)

# # Reshape to the required format for polylines/fillPoly
# pts = pts.reshape((-1, 1, 2))

# # Draw the filled diamond on the mask
# cv2.fillPoly(target_mask, [pts], 1)

cv2.imshow("target mask", target_mask)

print("Using just the edges of the mask:")
M = icp_2d.icp(source_mask, target_mask)
print("Transformation Matrix:")
print(M)
print()
print("Using entire mask:")
M = icp_2d.icp(source_mask, target_mask, edges_only=False)
print("Transformation Matrix:")
print(M)

cv2.waitKey(0)
cv2.destroyAllWindows()