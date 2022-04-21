# 
# after a lot of trial and errors :
# 3 ways to use tesseract :
# 1. calling the shell tesseract command via subprocess.run
# 2. calling pytesseract on the original image
# 3. calling pytesseract on the original image which has been extended with a row of white pixel on the top and on the bottom 
# and then we can apply those 3 methods on either the original image or on an optimised image 
# (for instance after having "eroded" the characters)
# the last one (method 3 on eroded) seems to work the most often, but not always

# Import packages
from audioop import add
import cv2
import numpy as np
import subprocess
import os
import pytesseract
import shlex
from event import create_event
import datetime

cv = cv2

# export DISPLAY=localhost:10.0

os.environ["DISPLAY"] = "localhost:10.0"
print(os.environ["DISPLAY"])

def get_cam_footage(basename):
    """
    get 1 second of video from the chalet Webcam and put it in <basename>.h264
    """
    process = subprocess.run(
        ['openRTSP', '-d', '1', '-V', '-F', f'{basename}-', 'rtsp://admin:123456@192.168.0.4/'], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True)
    # print("rc = ", process.returncode)
    # print("result = ", process.stdout)
    # err = process.stderr
    # # print("err = ", process.stderr)
    # $ cp chalet-video-H264-1 a.h264
    # $ vlc a.h264
    os.rename(f'{basename}-video-H264-1',f'{basename}.h264')

def get_snapshot(basename):
    """
    extract a snapshot from <basename>.h264 and put it in <basename>.jpg
    """
    # extra a picture from that video
    process = subprocess.run(
        ['ffmpeg', '-y', '-i', f'{basename}.h264', '-frames:v', '1', f'{basename}.jpg'], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True)
    # print("args = ", process.args)
    # print("rc = ", process.returncode)
    # print("result = ", process.stdout)
    # err = process.stderr
    # print("err = ", process.stderr)

def cropped_digits_img(filename):
    global interactive

    # read the snapshot
    img = cv2.imread(filename)
    # print(img.shape) # Print image shape
    # if interactive: cv2.imshow("original", img)

    # convert to grey only
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if interactive: cv2.imshow("greyed", img)

    #invert image (black becomes white)
    img = (255-img)
    # if interactive: cv2.imshow("greyed inverted", img)

    # Crop the image to focus onthe digits
    img = img[445:580, 620:1200]
    # cv2.imshow("cropped", img2)

    # thresholding
    ret, img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
    # print("cv2.THRESH_BINARY : ", cv2.THRESH_BINARY)

    # Display cropped image
    # if interactive: cv2.imshow("threshed", img)
    return img


def get_digits(img, options_list):
    # reads digits from picture
    # if interactive: cv2.imshow("cropped digits", img)
    temp_filename = "tmp_img.jpg"
    temp_output_filename = "tmp_output.txt"
    cv2.imwrite(temp_filename, img)

    # print("shell tesseract options: ", options_list)
    process = subprocess.run(
        #['tesseract', '-c', 'page_separator=', temp_filename, temp_output_filename] + options_list,
        ['tesseract', '-c', 'page_separator=', temp_filename, 'stdout' ] + options_list,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True)
    # print("args = ", process.args)
    # print("rc = ", process.returncode)
    # print("result = ", process.stdout)
    # err = process.stderr
    # print("err = ", process.stderr)
    return process.stdout.strip()


def check_digits(st):
    """
    checks if string st contains 3 digits, then a space, then 3 digits, and 
    return the corresponding int values (ph and Cl) if that's the case, or None otherwise
    """
    st = st.strip()
    pH = None
    Cl = None
    if len(st) == 7 and st[0:3].isnumeric and st[-3:].isnumeric:
        pH = int(st[0:3])/100.0
        Cl = int(st[-3:])
    return pH, Cl


def get_best_result(candidate_results, img):
    """
    results is an array of [label_str, result_str] (ex: ["tesseract optimised","743 423"])
    - create a list with only the valid results (format must be "999 999", after having removed any dot ("."))
    - if there are no result
        store the problematic image for later analysis in issues/noresult-<datetime>
        return None,None
    - if there are several valid results :
        - if all of them are the same :
            return pH,Cl
        - if there are different results :
            store the problematic image for later analysis in issues/ambiguous-<datetime>
            return first pH,Cl in the list
    """

    x = datetime.datetime.now()
    now_str = x.strftime("%Y-%m-%d_%H-%M-%S")

    # create a list of valid results only
    valid_results = []
    for c in candidate_results:
        # print(f'{c[0]:35}: {c[1]}')
        st = c[1].strip()
        # get rid of decimal dot (rare, but sometimes they are recognised by OCR)
        st = st.replace(".","")
        if len(st) == 7 and st[0:3].isnumeric and st[-3:].isnumeric:
            pH = int(st[0:3])/100.0
            Cl = int(st[-3:])
            # check the read figures make sense (sometimes a "7" is read as a "1" by tesseract)
            if pH > 3 and Cl > 300:
                valid_results.append(st)
    
    #remove duplicates from list of valid results
    valid_results = list(dict.fromkeys(valid_results))
   
    if len(valid_results) == 0:
        # no valid results; store image for later analysis
        pH = None
        Cl = None
        print("No valid results !")
        # store image for later analysis :
        filename = "issues/noresult_" + now_str + ".jpg"
        cv2.imwrite(filename, img)
    else:
        # at least one valid result; first one is kept and returned
        st = valid_results[0]
        pH = int(st[0:3])/100.0
        Cl = int(st[-3:])
        # if there were more than 1 valid result
        if len(valid_results) > 1:
            all_res = ""
            for res in valid_results:
                if all_res == "":
                    all_res = res[0:3] + res[-3:]
                else:
                    all_res = all_res + "_" + res[0:3] + res[-3:]
            print("more than 1 valid result : ", all_res)
            # store image for later analysis :
            filename = "issues/ambiguous_" + all_res +"_" + now_str + ".jpg"
            cv2.imwrite(filename, img)
    
    return pH, Cl


def optimise_img(img):
    """
    optimise the passed image by various methods, for instance by eroding the borders of the characters
    """

    # invert the image (because the erosion method I know work with white chars on black background images)
    # I am sure it can be made much better ;-)

    img = 255 - img
    # if interactive: cv2.imshow("inverted image", img)
 
    kernel = np.ones((5,5),np.uint8)

    # kernel = np.array( [    
    #     [ 0, 0, 0, 0, 0 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 0, 0, 0, 0, 0 ]
    #     ],np.uint8)

    # kernel = np.array( [    
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ]
    #     ])

    img = cv2.erode(img,kernel,iterations = 2)
    cv2.imwrite("erosion.jpg", img)
    # if interactive: cv2.imshow("erosion", img)  
    
    # invert the image again, since done at the beginning
    img = 255 - img

    return img


def explain_tesseract(img, title, options_str):
    """
    explains the tesseract way of analysing this image by having boxes drawn around the characters
    """
       
    r,c = img.shape
    nb_lines = 40
    additional_lines = np.full((nb_lines,c),255, dtype=np.uint8)
    
    # adding a blank rectangle above the image
    img2 = np.append(img,additional_lines,axis= 0)
    img2 = np.append(additional_lines,img2,axis= 0)

    # if interactive: cv2.imshow("img extended", img2)

    hImg,wImg = img.shape
    # print("pytesseract options", options_str)
    myres = pytesseract.image_to_string(img,config=options_str).strip()
    candidate_results.append([f'{title} (orig size)',myres])
    #if interactive: print("pytesseract (orig): ", myres)
    myres2 = pytesseract.image_to_string(img2,config=options_str).strip()
    candidate_results.append([f'{title} (extended)',myres2])
    #if interactive: print("pytesseract (extended): ", myres2)
    
    boxes = pytesseract.image_to_boxes(img,config=options_str)

    for b in boxes.splitlines():
        #print(b)
        b = b.split(' ')
        # print(b)
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(img2,(x,hImg-y+nb_lines),(w,hImg-h+nb_lines),(0,255,0),2)
        cv2.putText(img2,b[0],(x,hImg-y+25+nb_lines),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    #if interactive: cv2.imshow('Image with boxes', img2)
    # cv2.waitKey(0)


def check_pool():
    global candidate_results
    global interactive

    candidate_results = []

    # NB : shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_str="--psm 13 -c tessedit_char_whitelist='.0123456789 '"
    #options_str="--psm 6 -c tessedit_char_whitelist='.0123456789 '"
    # shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_list = shlex.split(options_str)

    basename = "chalet"
    get_cam_footage(basename)
    get_snapshot(basename)
    
    debug = False
    if debug:
        filename = "threshed_chalet1.jpg"
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        filename = basename+'.jpg'
        img = cropped_digits_img(filename)

    #if interactive: cv2.imshow("cropped digits", img); cv2.waitKey
    
    res1 = get_digits(img, options_list)
    candidate_results.append(["tess. not optimised",res1])
    #if interactive: print("tesseract not optimised : ",res1)
    #if interactive: cv2.imshow("not optimised", img)
    explain_tesseract(img, "pytess. not optimised", options_str)

    img = optimise_img(img)
    #if interactive: cv2.imshow("optimized", img)
    #print("")

    res2 = get_digits(img, options_list)
    candidate_results.append(["tess. optimised",res2])
    #if interactive: print("tesseract  optimised : ",res1)
    #if interactive: cv2.imshow("optimised", img)
    explain_tesseract(img, "pytess. optimised", options_str)
    #print("")
    
    pH,Cl = get_best_result(candidate_results, img)
    # pH,Cl = check_digits(res1)

    if pH != None:
        create_event("pool_pH",str(pH))

    if Cl != None:
        create_event("pool_Cl",str(Cl))

    if interactive:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pH, Cl


def main():
    pH,Cl = check_pool()
    if pH != None:
        print(f'pH : {pH} - Cl : {Cl}')
    else:
        print("Coudln't read the figures !")

interactive = False

if __name__ == '__main__':
    interactive = True
    main()
