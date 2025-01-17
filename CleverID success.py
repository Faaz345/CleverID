import cv2
import numpy as np
import pytesseract
import json
import webbrowser
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

template = cv2.imread('id_card_template.jpg', 0)
logo_template = cv2.imread('logo_template.jpg', 0)
h, w = template.shape

orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
kp_template, des_template = orb.detectAndCompute(template, None)
kp_logo_template, des_logo_template = orb.detectAndCompute(logo_template, None)

if des_template is None or len(des_template) == 0:
    raise ValueError("No descriptors found for the ID card template.")
if des_logo_template is None or len(des_logo_template) == 0:
    raise ValueError("No descriptors found for the logo template.")

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture(0)
card_detected = False
logo_detected = False
website_opened = False

with open('credentials.json', 'r') as file:
    students_credentials = json.load(file)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not card_detected:
        for scale in np.linspace(0.5, 1.5, 5):
            template_resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            h_temp, w_temp = template_resized.shape
            if h_temp > gray_frame.shape[0] or w_temp > gray_frame.shape[1]:
                continue

            res = cv2.matchTemplate(gray_frame, template_resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            print(f"Template Matching Confidence (scale {scale}): {max_val}")
            threshold = 0.6

            if max_val >= threshold:
                top_left = max_loc
                bottom_right = (top_left[0] + w_temp, top_left[1] + h_temp)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                card_region = gray_frame[top_left[1]:top_left[1] + h_temp, top_left[0]:top_left[0] + w_temp]
                kp_frame, des_frame = orb.detectAndCompute(card_region, None)

                if des_frame is not None and len(des_frame) > 0:
                    matches = flann.knnMatch(des_template, des_frame, k=2)
                    good_matches = []
                    for match in matches:
                        if len(match) == 2:
                            m, n = match
                            if m.distance < 0.6 * n.distance:
                                good_matches.append(m)

                    if len(good_matches) > 10:
                        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        if M is not None:
                            pts = np.float32([[0, 0], [0, h_temp], [w_temp, h_temp], [w_temp, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            dst += np.float32([top_left])
                            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                            student_name = pytesseract.image_to_string(card_region, config='--psm 7').strip()
                            print(f"Detected Student Name: {student_name}")
                            matching_credentials = [cred for cred in students_credentials if student_name.lower() in cred['name'].lower()]

                            if matching_credentials:
                                print(f"Found matching credentials: {matching_credentials[0]}")
                                username = matching_credentials[0]['username']
                                password = matching_credentials[0]['password']
                                card_detected = True
                            else:
                                print("No matching credentials found.")
                                card_detected = False

    if card_detected and not logo_detected:
        logo_res = cv2.matchTemplate(gray_frame, logo_template, cv2.TM_CCOEFF_NORMED)
        _, max_val_logo, _, max_loc_logo = cv2.minMaxLoc(logo_res)
        print(f"Logo Matching Confidence: {max_val_logo}")
        logo_threshold = 0.7

        if max_val_logo >= logo_threshold:
            top_left_logo = max_loc_logo
            bottom_right_logo = (top_left_logo[0] + logo_template.shape[1], top_left_logo[1] + logo_template.shape[0])
            cv2.rectangle(frame, top_left_logo, bottom_right_logo, (255, 0, 0), 2)
            logo_detected = True

            if not website_opened:
                webbrowser.open('https://s.amizone.net/')
                website_opened = True

                chrome_driver_path = 'C:\\webdriver\\chromedriver.exe'
                service = Service(chrome_driver_path)
                driver = webdriver.Chrome(service=service)
                driver.get('https://s.amizone.net/')

                username_field = driver.find_element(By.NAME, '_UserName')
                password_field = driver.find_element(By.NAME, '_Password')
                username_field.send_keys(username)
                password_field.send_keys(password)

                try:
                    login_button = driver.find_element(By.XPATH, '//button[text()="Login"]')
                    ActionChains(driver).move_to_element(login_button).click().perform()
                    print("Login button clicked.")
                except Exception as e:
                    print(f"Error clicking login button: {e}")

    cv2.imshow('Real-Time ID Card and Logo Detection with AR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
