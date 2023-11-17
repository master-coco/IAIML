#4 Display images
import matplotlib.pyplot as plt
from scipy import datasets

# Load the sample face image from scipy.datasets
face = datasets.ascent()

# Display the image
plt.imshow(face, cmap='gray')        # Use a grayscale colormap
plt.axis('off')                      # Turn off axis labels
plt.show()