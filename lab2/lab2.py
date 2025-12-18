import cv2
import matplotlib.pyplot as plt
import numpy as np


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8)
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


def main():
    # Ucitavamo sliku
    image = cv2.imread('coins.png')
    plt.imshow(image)
    plt.show()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()  # BGR u RGB za plt prikaz

    # Pretvaramo sliku u nijansu sive
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.show()

    # Binarna slika pomocu praga
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(thresh)
    plt.show()

    # Zatvaranje (morfoloska operacija)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing)
    plt.show()

    # Extraktovanje saturacije
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv_image)
    plt.show()
    saturation = hsv_image[:, :, 1]
    plt.imshow(saturation)
    plt.show()

    # Marker pomocu praga
    _, marker = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
    plt.imshow(marker)
    plt.show()

    # Otvaranje i zatvaranje za poboljsanje markera:
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    plt.imshow(marker)
    plt.show()
    reconstructed = morphological_reconstruction(marker, closing)
    plt.imshow(reconstructed, cmap='gray')
    plt.show()
    # Kombinovanje maski
    # final_mask = cv2.bitwise_and(closing, reconstructed)
    # plt.imshow(final_mask)
    # plt.show()

    # Prikaz originalne slike
    plt.imshow(image_rgb)
    plt.show()

    # Prikaz finalne maske u sivoj boji
    # plt.imshow(final_mask, cmap='gray')
    # plt.show()

    # Prikaz konacne slike gde je finalna maska primenjena na originalnu sliku
    result = cv2.bitwise_and(image, image, mask=reconstructed)
    plt.imshow(result)
    plt.show()
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result_rgb)
    plt.show()


main()