from pyzbar.pyzbar import decode
import cv2

# Read the image containing the barcode
image = cv2.imread("/opt/SinevaAGV/SinevaCodeAMR/src/vision/get_image_hk/src/img/saved_1111.png")

print("1111111")
# Decode the barcode
barcodes = decode(image)

print("22222222")
# Print the data contained in the barcode
for barcode in barcodes:
    print("3333")
    print(barcode.data.decode('utf-8'))