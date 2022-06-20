#
# after a lot of trial and errors :
# 3 ways to use tesseract :
# 1. calling the shell tesseract command via subprocess.run
# 2. calling pytesseract on the original image
# 3. calling pytesseract on the original image which has been extended with a row of white pixel on the top and on the bottom
# and then we can apply those 3 methods on either the original image or on an optimised image
# (for instance after having "eroded" the characters)
# the last one (method 3 on eroded) seems to work the most often, but not always

# sqlite3 /opt/db/mydatabase.db
# select time,text from events where categ='pool_ph' and time<"2022-04-21";
# select time,text from events where categ='pool_ph' and time<"2022-04-21";
# select text from events where categ="pool_night" and text like '7%';
# select time, text from events where categ="pool_ph" and text > "7.9"; # and time = "2022-04-22 15:10:44";
# select time, text from events where categ="pool_ph" and text < "4";
# delete from events where categ="pool_ph" and text < "4";


# Import packages
import sys
import subprocess
import os
import time
import datetime
import logging
import platform


import cv2
import numpy as np
import pytesseract
import shlex
# from audioop import add

from event import create_event
from event import read_where

import utils
import params

webcam = params.webcam
jpg_path = "jpg/"
issues_path = "issues/"


calib_ph_x = params.calib_ph_x
calib_ph_y = params.calib_ph_y
calib_ph_width = params.calib_ph_width
calib_ph_height = params.calib_ph_height

calib_cl_x = params.calib_cl_x
calib_cl_y = params.calib_cl_y
calib_cl_width = params.calib_cl_width
calib_cl_height = params.calib_cl_height

calib_status_ph_x = params.calib_status_ph_x
calib_status_ph_y = params.calib_status_ph_y
calib_status_ph_width = params.calib_status_ph_width
calib_status_ph_height = params.calib_status_ph_height

calib_status_cl_x = params.calib_status_cl_x
calib_status_cl_y = params.calib_status_cl_y
calib_status_cl_width = params.calib_status_cl_width
calib_status_cl_height = params.calib_status_cl_height

calib_status_pp_x = params.calib_status_pp_x
calib_status_pp_y = params.calib_status_pp_y
calib_status_pp_width = params.calib_status_pp_width
calib_status_pp_height = params.calib_status_pp_height

calib_pool_delai_x = params.calib_pool_delai_x
calib_pool_delai_y = params.calib_pool_delai_y
calib_pool_delai_width = params.calib_pool_delai_width
calib_pool_delai_height = params.calib_pool_delai_height

def set_display():
    """
    sets the environmnent $DISPLAY variable based on the value established the last time an Xserver 
    (typically MobaXterm) provided a local display and that value was stored in a local file
    """
    filename = "/home/toto/.display"
    if os.path.isfile(filename):
        f = open(filename, "r")
        display = f.readline().strip()
        print(f'display read in {filename} : {display}')
        f.close()
    else:
        display = "10"
    os.environ["DISPLAY"] = display


def get_cam_footage(basename, webcam):
    """
    get 1 second of video from the chalet Webcam and put it in <basename>.h264
    (this function should be identifical between pool and power)
    """
    process = subprocess.run(
        ['openRTSP', '-d', '1', '-V', '-F',
            f'{basename}-', webcam],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    # print("rc = ", process.returncode)
    # print("result = ", process.stdout)
    # err = process.stderr
    # # print("err = ", process.stderr)
    # $ cp chalet-video-H264-1 a.h264
    # $ vlc a.h264

    audio_file = f'{basename}-audio-PCMA-2'
    if os.path.isfile(audio_file):
        os.remove(audio_file)

    video_file = f'{basename}-video-H264-1'
    if os.path.isfile(video_file):
        footage_filename = f'{basename}.h264'
        os.rename(video_file, footage_filename)
    else:
        footage_filename = None
    return footage_filename


def get_snapshot(footage_filename):
    """
    extract a snapshot from <basename>.h264 and put it in <basename>.jpg
    (this function should be identifical between pool and power)
    """

    if footage_filename == None:
        return None
    basename_ext = os.path.basename(footage_filename)
    basename, ext = os.path.splitext(basename_ext)
    # extract a picture from that video
    process = subprocess.run(
        ['ffmpeg', '-y', '-i', f'{basename}.h264',
            '-frames:v', '1', f'{jpg_path}{basename}.jpg'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    my_stdout = process.stdout
    err = process.stderr
    # print("----args = ", process.args)
    # print("----rc = ", process.returncode)
    # print("----stdout = ", my_stdout)
    # print("----err = ", err)

    if os.path.isfile(f'{jpg_path}{basename}.jpg'):
        os.rename(f'{basename}.h264', f'{basename}.bak.h264')
        extracted_img_filename = f'{jpg_path}{basename}.jpg'
    else:
        extracted_img_filename = None

    return extracted_img_filename


def set_calibration(title, img, x, y, width, height):
    """"
    allows to move a rectangle on top of a given image and returns the x,y coordinates of the top left corner 
    of the rectangle
    (this function should be identifical between pool and power)
    """
    calib_step = params.calib_step  # calib_step (in pixels) to move the rectangle with each move
    mode = "P"   # P: arrows change position    S: arrows change size
    window_name = title
    flags = cv2.WINDOW_NORMAL & cv2.WINDOW_KEEPRATIO
    #flags = cv2.WINDOW_AUTOSIZE
    cv2.namedWindow(window_name, flags)
    cv2.resizeWindow(window_name, 1800, 1000)
    cv2.moveWindow(window_name, 10,10)

    while True:
        img2 = np.copy(img)
        mytext = f'({x},{y}) width:{width} height:{height} - calib_step (+/-) : {calib_step} - Mode:{mode}'
        cv2.putText(img=img2, text=mytext, org=(
            50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0), thickness=1)

        cv2.rectangle(img2, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.imshow(window_name, img2)
        k = cv2.waitKey(0)

        # print(k)

        mySystem = platform.system()
        if mySystem == "Windows":
            esc = 27
            up = 2490368
            down = 2621440
            left = 2424832
            right = 2555904
            plus = 43
            minus = 45
        if mySystem == "Linux":
            esc = 27
            up = 82
            down = 84
            left = 81
            right = 83
            plus = 43
            minus = 45

        if k == esc:
            break
        elif k == -1:  # normally -1 returned,so don't print it
            continue
        elif k == up:
            if mode == "P": y -= calib_step
            else: height -= calib_step
        elif k == down:
            if mode == "P": y += calib_step
            else: height += calib_step
        elif k == left:
            if mode == "P": x -= calib_step
            else: width -= calib_step
        elif k == right:
            if mode == "P": x += calib_step
            else: width += calib_step
        elif k == plus:
            calib_step += 1
        elif k == minus:
            calib_step -= 1
        elif k == ord("m"):
            mode = "P" if mode == "V" else "V"
        else:
            print(k)  # else print its value
    cv2.destroyAllWindows()
    return x, y, width, height


def test_best_threshold(kind, img, start, end, step):
    """
    displays the thresholded img with several values of threshold, starting from start to end with a step of step
    (this function should be identifical between pool and power)
    """
    # # # # testing the best threshold
    img_bck = np.copy(img)
    i = 0
    height = 200
    width = 500
    for t in range(start, end, step):
        img = np.copy(img_bck)
        _, img = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
        if interactive: 
            window_name = f"{kind} thresholded {t}"
            flags = cv2.WINDOW_NORMAL & cv2.WINDOW_KEEPRATIO
            #flags = cv2.WINDOW_AUTOSIZE
            cv2.namedWindow(window_name, flags)
            cv2.resizeWindow(window_name, width, height)
            cv2.moveWindow(window_name, 10, 10 + i*(height+10))

            cv2.imshow(window_name, img)
            
        i += 1
    img_bck = np.copy(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_best_thresholded_img(img, basename, kind, best, step):
    # Crop the image to focus on the interesting part
    # best : best value so far
    # step : step between 2 values of candidate best_threshold value
    
    start = best - 2*step
    end = best + 2*step + 1
        
    if kind == "ph":
        cropped_img = np.copy(img[calib_ph_y : calib_ph_y + calib_ph_height, calib_ph_x : calib_ph_x + calib_ph_width])
    elif kind == "cl":
        cropped_img = np.copy(img[calib_cl_y : calib_cl_y + calib_cl_height, calib_cl_x : calib_cl_x + calib_cl_width])
    elif kind == "status_ph":
        cropped_img = np.copy(img[calib_status_ph_y : calib_status_ph_y + calib_status_ph_height, calib_status_ph_x : calib_status_ph_x + calib_status_ph_width])
    elif kind == "status_cl":
        cropped_img = np.copy(img[calib_status_cl_y : calib_status_cl_y + calib_status_cl_height, calib_status_cl_x : calib_status_cl_x + calib_status_cl_width])
    elif kind == "status_pp":
        cropped_img = np.copy(img[calib_status_pp_y : calib_status_pp_y + calib_status_pp_height, calib_status_pp_x : calib_status_pp_x + calib_status_pp_width])
    elif kind == "pool_delai":
        cropped_img = np.copy(img[calib_pool_delai_y : calib_pool_delai_y + calib_pool_delai_height, calib_pool_delai_x : calib_pool_delai_x + calib_pool_delai_width])
    else:
        logging.error(f'unexpected kind : {kind} !!')
        
    # save a copy of the cropped img for later analysis
    img_name = basename + '_' + kind + '_cropped_gray1'
    write_gray_to_file(img_name, cropped_img)
    
    if params.calib_reset_threshold:
        test_best_threshold(kind, cropped_img, start, end, step)
    
    best_threshold = best

    # thresholding to get a black/white picture
    _, cropped_img = cv2.threshold(cropped_img, best_threshold, 255, cv2.THRESH_BINARY)

    # print("cv2.THRESH_BINARY : ", cv2.THRESH_BINARY)
    return cropped_img


def cropped_digits_pool_img(filename):
    """
    get the best cropped images from the image contained in file filename and return those images
    """
    global interactive

    # extract the image from the file filename

    basename_ext = os.path.basename(filename)
    basename, ext = os.path.splitext(basename_ext)

    # read the snapshot
    img = cv2.imread(filename)
    # logging.info(img.shape) # Print image shape
    # if interactive: cv2.imshow("original", img)

    # filename2 = "2tmp_pool_base.bak.jpg"
    # img = cv2.imread(filename2)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert to grey only
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if interactive: cv2.imshow("full greyed", img); # cv2.waitKey(0);

    # invert image (black becomes white)
    img = (255-img)
    # if interactive: cv2.imshow("greyed inverted", img)

    # img_reinverted = np.copy(img)
    # img_reinverted = (255-img_reinverted)
    # if interactive: cv2.imshow("greyed re-inverted", img_reinverted)

    # extract from img the various sub-images : 

    img_ph =         get_best_thresholded_img (img, basename, "ph",        params.threshold_ph,     10)
    img_cl =         get_best_thresholded_img (img, basename, "cl",        params.threshold_cl,     10)
    img_status_ph =  get_best_thresholded_img (img, basename, "status_ph", params.threshold_status_ph, 10)
    img_status_cl =  get_best_thresholded_img (img, basename, "status_cl", params.threshold_status_cl, 10)
    img_status_pp =  get_best_thresholded_img (img, basename, "status_pp", params.threshold_status_pp, 10)
    img_pool_delai = get_best_thresholded_img (img, basename, "pool_delai", params.threshold_pool_delai, 10)
    
    return img_ph, img_cl, img_status_ph, img_status_cl, img_status_pp, img_pool_delai


def get_OCR_string(img_name, img, options_list):
    """
    return the string read by Tesseract in img with option_list as tesseract options and img_name the label of the image
    (this function should be identifical between pool and power)

    """
    # reads digits from picture
    # if interactive: cv2.imshow("cropped digits", img)
    temp_filename = jpg_path + img_name + ".jpg"
    temp_output_filename = "tmp_output.txt"
    cv2.imwrite(temp_filename, img)

    # print("shell tesseract options: ", options_list)
    process = subprocess.run(
        #['tesseract', '-c', 'page_separator=', temp_filename, temp_output_filename] + options_list,
        ['tesseract', '-c', 'page_separator=',
            temp_filename, 'stdout'] + options_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    # print("args = ", process.args)
    # print("rc = ", process.returncode)
    # print("result = ", process.stdout)
    # err = process.stderr
    # print("err = ", process.stderr)
    return process.stdout.strip()


# def check_digits(st):
#     """
#     checks if string st contains 3 digits, then a space, then 3 digits, and 
#     return the corresponding int values (ph and cl) if that's the case, or None otherwise
#     """
#     st = st.strip()
#     ph = None
#     cl = None
#     if len(st) == 7 and st[0:3].isnumeric and st[-3:].isnumeric:
#         ph = int(st[0:3])/100.0
#         cl = int(st[-3:])
#     return ph, cl


def last_validated_value(categ):
    """
    get from the DB the last value of category categ
    """
    now1 = datetime.datetime.now()
    nowStr = now1.strftime('%Y-%m-%d %H:%M:%S')
    res = read_where(categ, 1, "")
    error = res["error"]
    short_error_msg = ""
    long_error_msg = ""
    validated_value = None

    if (error != ""):
        short_error_msg = "events server unresponsive !!!"
        long_error_msg = f"!!!! Error : could not read the last {categ} event - {error}"

    # check that date is OK
    if short_error_msg == "":
        event = res["events"][0]
        last_event_date = event["time"]
        try:
            last_event_Datetime = datetime.datetime.strptime(
                last_event_date, '%Y-%m-%d %H:%M:%S')
        except Exception as error:
            short_error_msg = "event date is not a valid date !!!"
            long_error_msg = f"date of last {categ} is not a date : {last_event_date} ! - {str(error)}"
            # logging.error(long_error_msg)

    # then check if the value we got is a valid float
    if short_error_msg == "":
        if event["text"].isnumeric:
            validated_value = float(event["text"])
        else:
            validated_value = None
            short_error_msg = "{categ} value is not a numeric value - {nowStr}"
            long_error_msg = "{categ} value is not a numeric value"

    if short_error_msg != "":
        user_name = params.mailer
        passwd = params.mailer_pw
        from_email = params.from_email
        to_email = params.to_email
        subject = short_error_msg + "- " + nowStr
        body = long_error_msg
        htmlbody = None
        myfilename = None
        utils.mySend(user_name, passwd, from_email,
                          to_email, subject, body, htmlbody, myfilename)

    return validated_value


def get_best_result(kind, candidate_results, img):
    """
    kind in {"ph", "cl"}
    results is an array of [label_str, result_str] (ex: ["tesseract optimised","743"])
    - create a list with only the valid results (format must be "999", after having removed any dot ("."))
    - if there are no result
        store the problematic image for later analysis in issues/noresult-<datetime>
        return None,None
    - if there are several valid results :
        - if all of them are the same, return that value
        - if there are different results :
            store the problematic image for later analysis in issues/ambiguous-<datetime>
            return the value having the less delta with the latest registered value in the db
    """

    x = datetime.datetime.now()
    now_str = x.strftime("%Y-%m-%d_%H-%M-%S")

    # create a list of valid results only
    valid_results = []

    for c in candidate_results:
        logging.info(f'{c[0]:35}: {c[1]}')
        st = c[1].strip()
        ph = None
        # get rid of decimal dot (rare, but sometimes they are recognised by OCR)
        st = st.replace(".", "")
        # there must be at least two numerical values separated by a space


        if (st[0:3].isnumeric()):
            if st[0:3].isnumeric(): candidate = int(st[0:3])
            # check the read figures make sense (sometimes a "7" is read as a "1" by tesseract)
            if kind == "ph":
                candidate = candidate*1.0/100
                if candidate > 2:
                    valid_results.append(candidate)
            if kind == "cl":
                if candidate > 300:
                    valid_results.append(candidate)


    # remove duplicates from list of valid results
    valid_results = list(dict.fromkeys(valid_results))

    if len(valid_results) == 0:
        # no valid results; store image for later analysis
        best_candidate = None
        print(f'No valid results for {kind} !')
        # store image for later analysis :
        filename = issues_path + now_str + "_" + kind + "_noresult.jpg"
        cv2.imwrite(filename, img)
    else:
        # at least one valid result; first one is kept, unless we find another one which is closer to the last_validated_val
        if kind == "ph":
            last_validated = last_validated_value("pool_ph")
        elif kind == "cl":
            last_validated = last_validated_value("pool_cl")

        best_candidate = valid_results[0]
        # if there were more than 1 valid result
        if len(valid_results) > 1:
            prev_delta = abs(best_candidate - last_validated)
            all_candidates = ""
            for candidate in valid_results:
                # find the delta between this candidate and the previously stored value in the DB
                delta = abs(candidate - last_validated)
                if delta < prev_delta:
                    best_candidate = candidate
                    prev_delta = delta
                    
                # accumulate in all_candidates a string with all the candidate values, for later analysis
                if all_candidates == "":
                    all_candidates = str(candidate)
                else:
                    all_candidates = all_candidates + "_" + str(candidate)
            logging.info(f'more than 1 valid result for {kind} : {all_candidates}')
            # store image for later analysis :
            filename = issues_path + now_str + "_" + kind + "_ambiguous_" + all_candidates + ".jpg"
            cv2.imwrite(filename, img)

    logging.info("")
    return best_candidate


def get_best_result_ph_cl_old(candidate_results, img):
    """
    results is an array of [label_str, result_str] (ex: ["tesseract optimised","743 423"])
    - create a list with only the valid results (format must be "999 999", after having removed any dot ("."))
    - if there are no result
        store the problematic image for later analysis in issues/noresult-<datetime>
        return None,None
    - if there are several valid results :
        - if all of them are the same :
            return ph,cl
        - if there are different results :
            store the problematic image for later analysis in issues/ambiguous-<datetime>
            return first ph,cl in the list
    """

    x = datetime.datetime.now()
    now_str = x.strftime("%Y-%m-%d_%H-%M-%S")

    # create a list of valid results only
    # valid_results = []
    valid_results_ph = []
    valid_results_cl = []
    last_validated_val_ph = last_validated_value("pool_ph")
    last_validated_val_cl = last_validated_value("pool_cl")

    for c in candidate_results:
        logging.info(f'{c[0]:35}: {c[1]}')
        st = c[1].strip()
        # get rid of decimal dot (rare, but sometimes they are recognised by OCR)
        ph = None
        cl = None
        st = st.replace(".", "")
        # there must be at least two numerical values separated by a space
        if st.find(" ") != -1 and (st[0:3].isnumeric() or st[-3:].isnumeric()):
            # print(f"st[0:3]: {st[0:3]} - st[0:3].isnumeric() : {st[0:3].isnumeric()}")
            # print(f"st[-3:]: {st[-3:]} - st[-3:].isnumeric() : {st[-3:].isnumeric()}")
            if st[0:3].isnumeric(): ph = int(st[0:3])/100.0
            if st[-3:].isnumeric(): cl = int(st[-3:])
            # check the read figures make sense (sometimes a "7" is read as a "1" by tesseract)
            # if ph > 3 and cl > 300:
            #     valid_results.append(st)
            if ph != None and ph > 4 :
                valid_results_ph.append(ph)
            if cl != None and cl > 400:
                valid_results_cl.append(cl)


    # remove duplicates from list of valid results for ph
    valid_results_ph = list(dict.fromkeys(valid_results_ph))

    # remove duplicates from list of valid results for cl
    valid_results_cl = list(dict.fromkeys(valid_results_cl))

    if len(valid_results_ph) == 0:
        # no valid results; store image for later analysis
        best_candidate_ph = None
        print("No valid results for ph !")
        # store image for later analysis :
        filename = issues_path + "noresult_ph_" + now_str + ".jpg"
        cv2.imwrite(filename, img)
    else:
        # at least one valid result; first one is kept, unless we find another one which is closer to the last_validated_val
        last_validated_ph = last_validated_value("pool_ph")
        last_validated_cl = last_validated_value("pool_cl")
        best_candidate_ph = valid_results_ph[0]
        # if there were more than 1 valid result
        if len(valid_results_ph) > 1:
            prev_delta = abs(best_candidate_ph - last_validated_ph)
            all_candidates_ph = ""
            for candidate in valid_results_ph:
                # find the delta between this candidate and the previously stored value in the DB
                delta = abs(candidate - last_validated_ph)
                if delta < prev_delta:
                    best_candidate_ph = candidate
                    prev_delta = delta
                    
                # accumulate in all_candidates a string with all the candidate values, for later analysis
                if all_candidates_ph == "":
                    all_candidates_ph = str(candidate)
                else:
                    all_candidates_ph = all_candidates_ph + "_" + str(candidate)
            logging.info(f'more than 1 valid result for cl : {all_candidates_ph}')
            # store image for later analysis :
            filename = issues_path + now_str + "_ambiguous_ph_" + all_candidates_ph + ".jpg"
            cv2.imwrite(filename, img)

    if len(valid_results_cl) == 0:
        # no valid results; store image for later analysis
        best_candidate_cl = None
        print("No valid results for cl !")
        # store image for later analysis :
        filename = issues_path + now_str + "_noresult_cl.jpg"
        cv2.imwrite(filename, img)
    else:
        # at least one valid result for cl; first one is kept and returned
        best_candidate_cl = valid_results_cl[0]
        if len(valid_results_cl) > 1:
            all_candidates_cl = ""
            for candidate in valid_results_cl:
                if all_candidates_cl == "":
                    all_candidates_cl = str(candidate)
                else:
                    all_candidates_cl = all_candidates_cl + "_" + str(candidate)
            print("more than 1 valid result for best_candidate_cl : ", all_candidates_cl)
            # store image for later analysis :
            filename = issues_path + now_str + "_ambiguous_cl_" + all_candidates_cl + ".jpg"
            cv2.imwrite(filename, img)

    logging.info("")
    return best_candidate_ph, best_candidate_cl


def get_best_result_status(candidate_results, img):
    """
    result is the non-tuple (ph, cl)
    algorithm :
    (label_str, result_str) 
    - create a list with only the valid results (format must be "999 999", after having removed any dot ("."))
        (ex: (["tesseract optimised","743 423"])
    - if there are no result
        store the problematic image for later analysis in issues/noresult-<datetime>
        return None,None
    - if there are several valid results :
        - if all of them are the same :
            return ph,cl
        - if there are different results :
            store the problematic image for later analysis in issues/ambiguous-<datetime>
            return first ph,cl in the list
    """

    x = datetime.datetime.now()
    now_str = x.strftime("%Y-%m-%d_%H-%M-%S")

    # create a list of valid results only
    valid_results = []
    best_result = ""
    
    for c in candidate_results:
        logging.info(f'{c[0]:35}: {c[1]}')
        # st = c[1].strip()
        # result = None
        # st = st.replace(".", "")
        # # there must be at least two numerical values separated by a space
        # if st.find(" ") != -1 and (st[0:3].isnumeric() or st[-3:].isnumeric()):
        #     # print(f"st[0:3]: {st[0:3]} - st[0:3].isnumeric() : {st[0:3].isnumeric()}")
        #     # print(f"st[-3:]: {st[-3:]} - st[-3:].isnumeric() : {st[-3:].isnumeric()}")
        #     if st[0:3].isnumeric(): ph = int(st[0:3])/100.0
        #     if st[-3:].isnumeric(): cl = int(st[-3:])
        #     # check the read figures make sense (sometimes a "7" is read as a "1" by tesseract)
        #     # if ph > 3 and cl > 300:
        #     #     valid_results.append(st)
        #     if ph != None and ph > 3 :
        #         valid_results_ph.append(ph)
        #     if cl != None and cl > 300:
        #         valid_results_cl.append(cl)


    # # remove duplicates from list of valid results for ph
    # valid_results_ph = list(dict.fromkeys(valid_results_ph))

    # # remove duplicates from list of valid results for cl
    # valid_results_cl = list(dict.fromkeys(valid_results_cl))

    # issues_path = "issues/"
    # if not os.path.isdir(issues_path):
    #     os.mkdir(issues_path)

    # if len(valid_results_ph) == 0:
    #     # no valid results; store image for later analysis
    #     best_candidate_ph = None
    #     print("No valid results for ph !")
    #     # store image for later analysis :
    #     filename = issues_path + "noresult_ph_" + now_str + ".jpg"
    #     cv2.imwrite(filename, img)
    # else:
    #     # at least one valid result; first one is kept, unless we find another one which is closer to the last_validated_val
    #     last_validated_ph = last_validated_value("pool_ph")
    #     last_validated_cl = last_validated_value("pool_cl")
    #     best_candidate_ph = valid_results_ph[0]
    #     # if there were more than 1 valid result
    #     if len(valid_results_ph) > 1:
    #         prev_delta = abs(best_candidate_ph - last_validated_ph)
    #         all_candidates_ph = ""
    #         for candidate in valid_results_ph:
    #             # find the delta between this candidate and the previously stored value in the DB
    #             delta = abs(candidate - last_validated_ph)
    #             if delta < prev_delta:
    #                 best_candidate_ph = candidate
    #                 prev_delta = delta
                    
    #             # accumulate in all_candidates a string with all the candidate values, for later analysis
    #             if all_candidates_ph == "":
    #                 all_candidates_ph = str(candidate)
    #             else:
    #                 all_candidates_ph = all_candidates_ph + "_" + str(candidate)
    #         logging.info(f'more than 1 valid result for cl : {all_candidates_ph}')
    #         # store image for later analysis :
    #         filename = issues_path + "ambiguous_ph_" + all_candidates_ph + "_" + now_str + ".jpg"
    #         cv2.imwrite(filename, img)

    # if len(valid_results_cl) == 0:
    #     # no valid results; store image for later analysis
    #     best_candidate_cl = None
    #     print("No valid results for cl !")
    #     # store image for later analysis :
    #     filename = issues_path + "noresult_cl_" + now_str + ".jpg"
    #     cv2.imwrite(filename, img)
    # else:
    #     # at least one valid result for cl; first one is kept and returned
    #     best_candidate_cl = valid_results_cl[0]
    #     if len(valid_results_cl) > 1:
    #         all_candidates_cl = ""
    #         for candidate in valid_results_cl:
    #             if all_candidates_cl == "":
    #                 all_candidates_cl = str(candidate)
    #             else:
    #                 all_candidates_cl = all_candidates_cl + "_" + str(candidate)
    #         print("more than 1 valid result for best_candidate_cl : ", all_candidates_cl)
    #         # store image for later analysis :
    #         filename = issues_path + "ambiguous_cl_" + all_candidates_cl + "_" + now_str + ".jpg"
    #         cv2.imwrite(filename, img)

    logging.info("")
    return best_result


def optimise_img(img):
    """
    optimise the passed image by various methods, for instance by eroding the borders of the characters
    """

    # invert the image (because the erosion method I know work with white chars on black background images)
    # I am sure it can be made much better ;-)

    img = 255 - img
    # if interactive: cv2.imshow("inverted image", img)

    kernel = np.ones((5, 5), np.uint8)

    # kernel = np.array( [
    #     [ 0, 0, 0, 0, 0 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 0, 0, 0, 0, 0 ]
    #     ],np.uint8)

    img = cv2.erode(img, kernel, iterations=2)
    # cv2.imwrite("base_eroded.jpg", img)
    # if interactive: cv2.imshow("eroded", img)

    # invert the image again, since done at the beginning
    img = 255 - img

    return img


def show_boxes(kind, title, img_orig, options_str, nb_lines):
    
    if not kind in params.show_boxes: return
    
    
    img = np.copy(img_orig)
    print("orig shape : ", img.shape)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    print("converted shape : ", img.shape)
    
    boxes = pytesseract.image_to_boxes(img, config=options_str)
    hImg, wImg, channels = img.shape


    r, c, _ = img.shape

    if nb_lines > 0:
        additional_lines = np.full((nb_lines, c, 3), 255, dtype=np.uint8)
        # adding a blank rectangle above and below the image
        img = np.append(img, additional_lines, axis=0)
        img = np.append(additional_lines, img, axis=0)


    for b in boxes.splitlines():
        # print(b)
        b = b.split(' ')
        # print(b)
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        
        cv2.rectangle(img, (x, hImg-y+nb_lines),
                      (w, hImg-h+nb_lines), (0, 255, 0), 1)
        cv2.putText(img, b[0], (x, hImg-y+25+nb_lines),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)


    window_name = f'{kind}-{title}'
    flags = cv2.WINDOW_NORMAL & cv2.WINDOW_KEEPRATIO
    #flags = cv2.WINDOW_AUTOSIZE
    cv2.namedWindow(window_name, flags)
    cv2.imshow(window_name, img)
    cv2.resizeWindow(window_name, 1800, 1000)
    cv2.moveWindow(window_name, 10,10)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #if interactive: cv2.imshow('Image with boxes', img2)
    # cv2.waitKey(0)


def explain_tesseract(img, kind, title, options_str,candidate_results):
    """
    explains the tesseract way of analysing this image by having boxes drawn around the characters
    """

    r, c = img.shape
    nb_lines = 40
    additional_lines = np.full((nb_lines, c), 255, dtype=np.uint8)

    # adding a blank rectangle above the image
    img2 = np.append(img, additional_lines, axis=0)
    img2 = np.append(additional_lines, img2, axis=0)

    myres = pytesseract.image_to_string(img, config=options_str).strip()
    candidate_results.append([f'{kind}-{title} (orig size)', myres])
    show_boxes(kind, f'{title} (orig size)', img, options_str, nb_lines)
    
    myres2 = pytesseract.image_to_string(img2, config=options_str).strip()
    candidate_results.append([f'{kind}-{title} (extended size)', myres2])
    show_boxes(kind, f'{title} (extended size)', img2, options_str, 0)


def write_gray_to_file(img_name, img):
    """
    takes a gray image and write it to disk
    """
    img_to_save = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(jpg_path + img_name + ".jpg", img_to_save)

def write_colour_to_file(img_name, img):
    """
    takes a colour image and write it to disk
    """
    img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(jpg_path + img_name + ".jpg", img_to_save)

def collect_candidate_results(img, kind, basename):
    """
    extract from img, of given kind ("day", "night", etc) all the possible candidates as 
    read string of numerical digits
    """
    # NB : shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_str = "--psm 13 -c tessedit_char_whitelist='.0123456789 '"
    #options_str="--psm 6 -c tessedit_char_whitelist='.0123456789 '"
    # shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_list = shlex.split(options_str)

    candidate_results = []
    img_name = basename+'_' + kind + '_cropped'
    #if interactive: cv2.imshow(img_name, img)
    # save a copy of this plain image for later analysis
    write_gray_to_file(img_name, img)

    # # read it again to check
    # img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if interactive: cv2.imshow("cropped digits", img); cv2.waitKey

    # extract the figures from this plain image
    res1 = get_OCR_string(img_name, img, options_list)
    candidate_results.append([kind + " tess. not optimised", res1])
    # if interactive: print("tesseract not optimised : ",res1)
    # if interactive: cv2.imshow("not optimised", img)
    explain_tesseract(img, kind, "pytess. not optimised",
                      options_str, candidate_results)

    # try to optimise the image
    img = optimise_img(img)
    img_name = basename+'_' + kind + '_optimised'
    #if interactive: cv2.imshow(img_name, img)
    # save a copy of this plain image for later analysis
    write_gray_to_file(img_name, img)

    # extract the figures from this optimised image
    res2 = get_OCR_string(img_name, img, options_list)
    candidate_results.append([kind + " tess. optimised", res2])
    # if interactive: print("tesseract  optimised : ",res1)
    # if interactive: cv2.imshow("optimised", img)
    explain_tesseract(img, kind, "pytess. optimised",
                      options_str, candidate_results)

    return candidate_results

def collect_candidate_results_status(img, kind, basename):
    """
    extract from img all the possible candidates as ph/cl status
    """
    # # NB : shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    # options_str = "--psm 13 -c tessedit_char_whitelist='.0123456789 '"
    options_str = ""
    # #options_str="--psm 6 -c tessedit_char_whitelist='.0123456789 '"
    
    # shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_list = shlex.split(options_str)

    candidate_results = []
    img_name = basename+'_' + kind + '_cropped'
    #if interactive: cv2.imshow(img_name, img)
    # save a copy of this plain image for later analysis
    write_gray_to_file(img_name, img)

    # # read it again to check
    # img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if interactive: cv2.imshow("cropped digits", img); cv2.waitKey

    # extract the figures from this plain image
    res1 = get_OCR_string(img_name, img, options_list)
    candidate_results.append([kind + " tess. not optimised", res1])
    # if interactive: print("tesseract not optimised : ",res1)
    # if interactive: cv2.imshow("not optimised", img)
    explain_tesseract(img, kind, "pytess. not optimised",
                      options_str, candidate_results)

    # try to optimise the image
    img = optimise_img(img)
    img_name = basename+'_' + kind + '_optimised'
    #if interactive: cv2.imshow(img_name, img)
    # save a copy of this plain image for later analysis
    write_gray_to_file(img_name, img)

    # extract the figures from this optimised image
    res2 = get_OCR_string(img_name, img, options_list)
    candidate_results.append([kind + " tess. optimised", res2])
    # if interactive: print("tesseract  optimised : ",res1)
    # if interactive: cv2.imshow("optimised", img)
    explain_tesseract(img, kind, "pytess. optimised",
                      options_str, candidate_results)

    return candidate_results


def display_candidate_results(candidate_results):
    for c in candidate_results:
        if interactive:
            print(f'{c[0]:35}: {c[1]}')


def check_pool():
    global candidate_results
    global interactive

    # NB : shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_str = "--psm 13 -c tessedit_char_whitelist='.0123456789 '"
    #options_str="--psm 6 -c tessedit_char_whitelist='.0123456789 '"
    # shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_list = shlex.split(options_str)

    basename = "pool_base"


    # if debug:
    #     filename = "threshed_chalet1.jpg"
    #     img = cv2.imread(filename)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else:
    #     footage_filename = get_cam_footage(basename, webcam)
    #     img_filename = get_snapshot(footage_filename)
    #     if img_filename != None and os.path.isfile(img_filename):
    #         img = cropped_digits_pool_img(img_filename)
    #         img_filename_bak = "tmp_"+basename+'.bak.jpg'
    #         os.rename(img_filename, img_filename_bak)
    #     else:
    #         img = None


    successful = False
    i = 1
    max_iteration = 10
    while not successful and i <= max_iteration:
        footage_filename = get_cam_footage("tmp_"+basename, webcam)
        if footage_filename != None:
            successful = get_snapshot("tmp_"+basename)
        else:
            logging.error(f'iter {i} : failed to get footage')
        if not successful:
            logging.error(f'iter {i} : failed to get snapshot out of footage')
        i += 1

    if successful:
        debug = False
        if debug:
            filename = jpg_path + "threshed_chalet1.jpg"
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            filename = jpg_path + "tmp_"+basename+'.jpg'
            filename_bak = jpg_path + "tmp_"+basename+'.bak.jpg'
            img_ph, img_cl, img_status_ph, img_status_cl, img_status_pp , img_pool_delai = cropped_digits_pool_img(filename)
            os.rename(filename, filename_bak)

        if interactive:
            print("")

    #------------------------------------------
    # OCR of ph
    img = img_ph
    if isinstance(img,np.ndarray) and img.any() != None:
        candidate_results = collect_candidate_results(img, "pool_ph", basename)
        
        ph = get_best_result("ph", candidate_results, img)

        if ph != None:
            create_event("pool_ph", str(ph))

        if interactive:
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        ph = None

    #------------------------------------------
    # OCR of cl
    img = img_cl
    if isinstance(img,np.ndarray) and img.any() != None:
        candidate_results = collect_candidate_results(img, "pool_cl", basename)
        
        cl = get_best_result("cl", candidate_results, img)

        if cl != None:
            create_event("pool_cl", str(cl))

        if interactive:
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        cl = None

    #------------------------------------------
    # OCR of status_ph
    img = img_status_ph
    if isinstance(img,np.ndarray) and img.any() != None:
        candidate_results_status_ph = collect_candidate_results_status(img, "status_ph", basename)
        
        status_ph = get_best_result_status(candidate_results_status_ph, img)
        # ph,cl = check_digits(res1)

        if status_ph != None:
            create_event("pool_status_ph", status_ph)

        if interactive:
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        status_ph = None

    #------------------------------------------
    # OCR of status_cl
    img = img_status_cl
    if isinstance(img,np.ndarray) and img.any() != None:
        candidate_results_status_cl = collect_candidate_results_status(img, "status_cl", basename)
        
        status_cl = get_best_result_status(candidate_results_status_cl, img)
        # cl,cl = check_digits(res1)

        if status_cl != None:
            create_event("pool_status_cl", status_cl)

        if interactive:
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        status_cl = None

    #------------------------------------------
    # OCR of status_pp
    img = img_status_pp
    if isinstance(img,np.ndarray) and img.any() != None:
        candidate_results_status_p = collect_candidate_results_status(img, "status_pp", basename)
        
        status_pp = get_best_result_status(candidate_results_status_p, img)
        # cl,cl = check_digits(res1)

        if status_pp != None:
            create_event("pool_status_pp", status_pp)

        if interactive:
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        status_pp = None

    #------------------------------------------
    # OCR of pool_delai
    img = img_pool_delai
    if isinstance(img,np.ndarray) and img.any() != None:
        candidate_results_pool_delai = collect_candidate_results_status(img, "pool_delai", basename)
        
        pool_delai = get_best_result_status(candidate_results_pool_delai, img)
        # cl,cl = check_digits(res1)

        if pool_delai != None:
            create_event("pool_pool_delai", pool_delai)

        if interactive:
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        pool_delai = None



    return ph, cl, status_ph, status_cl, status_pp, pool_delai


def calibration_pool():
    global calib_ph_x, calib_ph_y, calib_ph_width, calib_ph_height
    global calib_cl_x, calib_cl_y, calib_cl_width, calib_cl_height
    global calib_status_ph_x, calib_status_ph_y, calib_status_ph_width, calib_status_ph_height
    global calib_status_cl_x, calib_status_cl_y, calib_status_cl_width, calib_status_cl_height
    global calib_status_pp_x, calib_status_pp_y, calib_status_pp_width, calib_status_pp_height
    global calib_pool_delai_x, calib_pool_delai_y, calib_pool_delai_width, calib_pool_delai_height
    basename = "pool_base"

    footage_filename = get_cam_footage(basename, webcam)
    img_filename = get_snapshot(footage_filename)
    img = None
    if img_filename != None and os.path.isfile(img_filename):
        img = cv2.imread(img_filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if isinstance(img,np.ndarray):
        
        title = "ph"        
        calib_ph_x, calib_ph_y, calib_ph_width, calib_ph_height = set_calibration(
            title, img, calib_ph_x, calib_ph_y, calib_ph_width, calib_ph_height)
        utils.replace_param("params.py", "calib_ph_x", calib_ph_x)
        utils.replace_param("params.py", "calib_ph_y", calib_ph_y)
        utils.replace_param("params.py", "calib_ph_width", calib_ph_width)
        utils.replace_param("params.py", "calib_ph_height", calib_ph_height)
        logging.info(
            f'x:{calib_ph_x}, y:{calib_ph_y}, width:{calib_ph_width}, height:{calib_ph_height}')

        title = "cl"        
        calib_cl_x, calib_cl_y, calib_cl_width, calib_cl_height = set_calibration(
            title, img, calib_cl_x, calib_cl_y, calib_cl_width, calib_cl_height)
        utils.replace_param("params.py", "calib_cl_x", calib_cl_x)
        utils.replace_param("params.py", "calib_cl_y", calib_cl_y)
        utils.replace_param("params.py", "calib_cl_width", calib_cl_width)
        utils.replace_param("params.py", "calib_cl_height", calib_cl_height)
        logging.info(
            f'x:{calib_cl_x}, y:{calib_cl_y}, width:{calib_cl_width}, height:{calib_cl_height}')

        title = "status_ph"        
        calib_status_ph_x, calib_status_ph_y, calib_status_ph_width, calib_status_ph_height = set_calibration(
            title, img, calib_status_ph_x, calib_status_ph_y, calib_status_ph_width, calib_status_ph_height)
        utils.replace_param("params.py", "calib_status_ph_x", calib_status_ph_x)
        utils.replace_param("params.py", "calib_status_ph_y", calib_status_ph_y)
        utils.replace_param("params.py", "calib_status_ph_width", calib_status_ph_width)
        utils.replace_param("params.py", "calib_status_ph_height", calib_status_ph_height)
        logging.info(
            f'x:{calib_status_ph_x}, y:{calib_status_ph_y}, width:{calib_status_ph_width}, height:{calib_status_ph_height}')

        title = "status_cl"        
        calib_status_cl_x, calib_status_cl_y, calib_status_cl_width, calib_status_cl_height = set_calibration(
            title, img, calib_status_cl_x, calib_status_cl_y, calib_status_cl_width, calib_status_cl_height)
        utils.replace_param("params.py", "calib_status_cl_x", calib_status_cl_x)
        utils.replace_param("params.py", "calib_status_cl_y", calib_status_cl_y)
        utils.replace_param("params.py", "calib_status_cl_width", calib_status_cl_width)
        utils.replace_param("params.py", "calib_status_cl_height", calib_status_cl_height)
        logging.info(
            f'x:{calib_status_cl_x}, y:{calib_status_cl_y}, width:{calib_status_cl_width}, height:{calib_status_cl_height}')

        title = "status_pp"        
        calib_status_pp_x, calib_status_pp_y, calib_status_pp_width, calib_status_pp_height = set_calibration(
            title, img, calib_status_pp_x, calib_status_pp_y, calib_status_pp_width, calib_status_pp_height)
        utils.replace_param("params.py", "calib_status_pp_x", calib_status_pp_x)
        utils.replace_param("params.py", "calib_status_pp_y", calib_status_pp_y)
        utils.replace_param("params.py", "calib_status_pp_width", calib_status_pp_width)
        utils.replace_param("params.py", "calib_status_pp_height", calib_status_pp_height)
        logging.info(
            f'x:{calib_status_pp_x}, y:{calib_status_pp_y}, width:{calib_status_pp_width}, height:{calib_status_pp_height}')

        title = "pool_delai"        
        calib_pool_delai_x, calib_pool_delai_y, calib_pool_delai_width, calib_pool_delai_height = set_calibration(
            title, img, calib_pool_delai_x, calib_pool_delai_y, calib_pool_delai_width, calib_pool_delai_height)
        utils.replace_param("params.py", "calib_pool_delai_x", calib_pool_delai_x)
        utils.replace_param("params.py", "calib_pool_delai_y", calib_pool_delai_y)
        utils.replace_param("params.py", "calib_pool_delai_width", calib_pool_delai_width)
        utils.replace_param("params.py", "calib_pool_delai_height", calib_pool_delai_height)
        logging.info(
            f'x:{calib_pool_delai_x}, y:{calib_pool_delai_y}, width:{calib_pool_delai_width}, height:{calib_pool_delai_height}')


    else:
        logging.error("Cannot calibrate because didn't get an image")
    

def print_usage():
    print("Usage : ")
    print(" python pool.py         : get the pool figures and display them on stdout")
    # print(" python pool.py where [[[ categ ] nb ] date_from ]     : prints the most recent ps4 pools")  
    print(" python pool.py calib   : recalibrate the cropping of the image")  
    print(" python pool.py anythingelse : print this usage")


def check_necessary_dirs():
    """
    create the necessary folders if not yet exist
    """
    if not os.path.isdir(issues_path):
        os.mkdir(issues_path)

    if not os.path.isdir(jpg_path):
        os.mkdir(jpg_path)


def main():
    utils.init_logger('INFO')
    logging.info("-----------------------------------------------------")
    logging.info("Starting pool")

    # set display based on last value, just in case there is a needed calibration, threshold adjustment or tesseract box checking
    set_display()

    check_necessary_dirs()
    
    do_calibration = False
    nb_args = len(sys.argv)
    logging.info(f'Number of arguments: {nb_args} arguments.')
    logging.info(f'Argument List: {str(sys.argv)}')
    if nb_args == 2:
        arg1 = sys.argv[1]
        logging.info(f"arg1 = {arg1}")
        if arg1 == "calib":
            do_calibration = True
        else:
            print_usage()

    if do_calibration or params.do_calibration:
        calibration_pool()

    ph, cl, status_ph, status_cl, status_pp, pool_delai = check_pool()
    if ph != None or cl != None:
        logging.info(f'ph : {ph} - cl : {cl}')
    else:
        logging.info("Coudln't read the figures !")

    logging.info("Ending pool")
    utils.shutdown_logger()


calibration_requested = False
interactive = False

if __name__ == '__main__':
    import getpass
    username = getpass.getuser()
    interactive = (username == "toto")
    debug = False
    main()
