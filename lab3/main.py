import cv2
import numpy as np


def detect_and_match_features(img1, img2):
    """
    Funkcija koja pronalazi karakteristične tačke na dve slike i uparuje ih.
    """
    # 1. Konverzija u sivo (Grayscale)
    # Algoritmi za detekciju tačaka rade brže i preciznije na jednokanalnim (crno-belim) slikama.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. Inicijalizacija SIFT detektora
    # SIFT (Scale-Invariant Feature Transform) pronalazi tačke koje su otporne na
    # promene veličine (zoom) i rotaciju.
    sift = cv2.SIFT_create()

    # detectAndCompute vraća:
    # kp (Keypoints): Koordinate tačaka (x, y)
    # des (Descriptors): Matematički otisak okoline te tačke (niz brojeva)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 3. Uparivanje tačaka (Matching)
    # Koristimo Brute-Force Matcher koji poredi svaki deskriptor sa prve slike
    # sa svim deskriptorima druge slike.
    bf = cv2.BFMatcher()

    # knnMatch vraća k=2 najbolja meča za svaku tačku (najbolji i drugi najbolji).
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    # 4. Filtriranje loših mečeva (Lowe's Ratio Test)
    # Ako je najbolji meč (m) mnogo bliži od drugog najboljeg (n), to je dobar meč.
    # Ako su slični (npr. plavo nebo vs plavo nebo), odbacujemo ih.
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return kp1, kp2, good


def warp_two_images(img_base, img_to_warp):
    """
    Spaja dve slike.
    img_base: Slika koja stoji "mirno" (referenca).
    img_to_warp: Slika koja se deformiše (krivi) da se uklopi u base.
    """
    # Prvo nalazimo zajedničke tačke
    kp_base, kp_warp, good_matches = detect_and_match_features(img_base, img_to_warp)

    # Ako nemamo bar 4 tačke, ne možemo izračunati perspektivu (homografiju)
    if len(good_matches) < 4:
        print("Nema dovoljno mečeva za homografiju.")
        return None

    # Izdvajamo koordinate tačaka iz objekata mečeva
    # queryIdx -> tačke sa img_base
    # trainIdx -> tačke sa img_to_warp
    pts_base = np.float32([kp_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_warp = np.float32([kp_warp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Izračunavanje Homografije (matrice H)
    # H je matrica 3x3 koja opisuje kako treba iskriviti pts_warp da se poklope sa pts_base.
    # RANSAC je algoritam koji ignoriše "šum" (pogrešne mečeve).
    H, status = cv2.findHomography(pts_warp, pts_base, cv2.RANSAC, 5.0)

    # --- Računanje veličine novog platna (Canvas) ---
    # Moramo znati koliko velika će biti finalna slika jer warpovana slika može otići
    # u minus koordinate ili daleko u plus.

    h1, w1 = img_base.shape[:2]
    h2, w2 = img_to_warp.shape[:2]

    # Uzimamo 4 ugla slike koju warpujemo (img_to_warp)
    pts_corners_warp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Primenjujemo matricu H na te uglove da vidimo gde će završiti nakon transformacije
    pts_corners_warp_trans = cv2.perspectiveTransform(pts_corners_warp, H)

    # Uzimamo 4 ugla bazne slike (ona se ne pomera, pa su uglovi isti)
    pts_corners_base = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # Spajamo sve tačke (bazne i nove warpovane) u jednu listu
    all_pts = np.concatenate((pts_corners_base, pts_corners_warp_trans), axis=0)

    # Nalazimo minimalne i maksimalne X i Y koordinate (granice nove slike)
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    # Ako slika ode u minus (npr. x_min = -100), moramo sve pomeriti za +100
    translation_dist = [-x_min, -y_min]

    # Kreiramo matricu translacije (pomera sliku da stane u vidljivo polje)
    # dtype=np.float32 je obavezan da bi množenje matrica radilo ispravno
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]], dtype=np.float32)

    # Konačne dimenzije nove slike
    final_w = x_max - x_min
    final_h = y_max - y_min

    # --- Warping i Spajanje ---

    # 1. Warpujemo drugu sliku
    # Koristimo H_translation.dot(H) -> Prvo primeni H (iskrivi), pa H_translation (pomeri)
    output_img = cv2.warpPerspective(img_to_warp, H_translation.dot(H), (final_w, final_h))

    # 2. Warpujemo baznu sliku
    # Bazna slika se NE krivi, ali se mora pomeriti (translirati) ako je nova slika proširena u levo/gore
    base_transformed = cv2.warpPerspective(img_base, H_translation, (final_w, final_h))

    # 3. Lepljenje (Blending)
    # Pravimo masku gde postoje pikseli bazne slike i tu ih "lepimo" preko warpovane slike.
    # Ovo prekriva eventualne crne rupe ili preklapanja na mestu spoja.
    mask = (base_transformed > 0)
    output_img[mask] = base_transformed[mask]

    return output_img


# --- GLAVNI PROGRAM ---

# Učitavanje slika sa diska
img1 = cv2.imread('slika1.png')  # Leva slika
img2 = cv2.imread('slika2.png')  # Srednja slika
img3 = cv2.imread('slika3.png')  # Desna slika

# Provera da li su slike uspešno učitane
if img1 is None or img2 is None or img3 is None:
    print("Greška: Neke slike nisu pronađene. Proveri putanje i imena fajlova.")
else:
    print("Spajanje slike 2 i 3...")
    # PRVI KORAK: Spajamo Srednju i Desnu.
    # img2 je baza, img3 se krivi prema njoj.
    # Rezultat je panorama desne strane.
    panorama_right = warp_two_images(img2, img3)

    if panorama_right is not None:
        print("Spajanje rezultata sa slikom 1...")
        # DRUGI KORAK: Spajamo Levu sliku na prethodni rezultat.
        # Sada je 'panorama_right' nova baza, a img1 (leva) se krivi da se uklopi u nju.
        final_panorama = warp_two_images(panorama_right, img1)

        if final_panorama is not None:
            # Prikazujemo rezultat
           # cv2.imshow('Panorama', panorama_right)
            cv2.imshow('Finalna Panorama', final_panorama)
            # Čuvamo sliku na disk
            cv2.imwrite('panorama_final.jpg', final_panorama)
            print("Gotovo! Slika sačuvana kao 'panorama_final.jpg'")

            # Čekamo pritisak bilo kog tastera pre zatvaranja prozora
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Neuspešno spajanje leve slike.")
    else:
        print("Neuspešno spajanje desne slike.")