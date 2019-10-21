import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import threading
import datetime
import requests
import json
import sys
import re
import pprint

class Walkpass_Analyzer:
    def __init__(self):
        self.passengernum = 0
        self.numcount = 0
        self.buf = []
        self.moviebuf = []
        self.maskbuf = []
        self.framecount = 0
        self.cap = None
        self.BUFFLEN = 100
        self.zerotorelance = 10
        self.area_thresh = 5000
        self.cap_size = [240,360]
        # threading
        self.recthread = None
        self.lock = threading.Lock()
        self.url = ''
        # background seraching count
        self.bgcount = 0
        self.seturl()

    def seturl(self):
        with open('weburl.ini',"r") as f:
            self.url = f.readline()
        self.url = self.url[:-1]
    def setcap(self, capname):
        if re.match(r'\d{1}',capname)!=None:
            self.cap = cv2.VideoCapture(int(capname))
        else:
            self.cap = cv2.VideoCapture(capname)
        
    def releasecap(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def resetpassengernum(self):
        self.passengernum = 0

    def sendtoslack(self, text_):
        WEB_HOOK_URL = self.url
        print(self.url)
        response = requests.post(self.url, data=json.dumps({
            'text': text_,
            'username': 'Camera'
        }),headers={'Content-Type': 'application/json'})
        #print(response.json())

    def recresult(self):
        buf_ = None
        maskbuf_ = None
        if os.path.exists('rec_movs')==False:
            os.mkdir('rec_movs')
        with self.lock:
            buf_ = self.moviebuf.copy()
            maskbuf_ = self.maskbuf.copy()
        if len(buf_) > 0:
            h, w = buf_[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            dt_now = datetime.datetime.now()
            str_time = '{:0=4}'.format(dt_now.year) + '{:0=2}'.format(dt_now.month) + \
                '{:0=2}'.format(dt_now.day) + '{:0=2}'.format(dt_now.hour) + \
                '{:0=2}'.format(dt_now.minute) + \
                '{:0=2}'.format(dt_now.second)+'.avi'
            writer = cv2.VideoWriter(os.path.join(
                'rec_movs', str_time), fourcc, 20.0, (w, h))
            # position center of the non-zero pixels
            mu_x = []
            for i in range(len(buf_) - self.zerotorelance * 3, len(buf_)):
                M_ = cv2.moments(maskbuf_[i], False)
                mu_x.append(int(M_['m10']/M_['m00']))
                writer.write(buf_[i])
            writer.release()
            if mu_x[0] > mu_x[-1]:
                str_direction = 'R to L'
            else:
                str_direction = 'L to R'
            str_slack = '{:0=4}'.format(dt_now.year) + '/{:0=2}'.format(dt_now.month) + \
                '/{:0=2}'.format(dt_now.day) + '/{:0=2}'.format(dt_now.hour) + \
                ':{:0=2}'.format(dt_now.minute) + \
                ':{:0=2}'.format(dt_now.second)
            self.sendtoslack(
                'A person has just passed by! Time:' + str_slack + ', Direction: '+str_direction)

    def startcap(self, REC=False):
        # initialize buf
        for i in range(self.BUFFLEN):
            self.buf.append(0)
        # initialize frame counting
        zero_count = 0  # num of continuous zero
        non_zero_count = 0  # num of continuous-non zero

        # capture setting
        h_s, w_s = self.cap_size[0], self.cap_size[1]
        x = int(w_s/2)
        if REC == True:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(
                'Demo_out.mp4', fourcc, 30.0, (w_s * 2, h_s))
        if self.cap is not None:
            _, frame_pre = self.cap.read()
            if frame_pre is None:
                return
            else:
                frame_pre = cv2.resize(frame_pre, (w_s, h_s))
                frame_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)
            # start scanning
            passtime = 0
            wnum_sum = 0
            diff_graph = []
            timeframe = []
            framecount = 0
            while True:
                _, frame = self.cap.read()
                if frame is None:
                    break
                else:

                    frame = cv2.resize(frame, (w_s, h_s))
                    self.moviebuf.append(frame)
                    if(len(self.moviebuf) > self.BUFFLEN):
                        self.moviebuf = self.moviebuf[1:]
                    frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    timeframe.append(framecount)
                    mask = self.get_subscription(frame_g, frame_pre)
                    self.maskbuf.append(mask)
                    if(len(self.maskbuf) > self.BUFFLEN):
                        self.maskbuf = self.maskbuf[1:]
                    difnum = mask.sum()
                    diff_graph.append(difnum)
                    wnum = sum(mask[:, x])
                    self.buf.append(wnum)
                    if(len(self.buf) > self.BUFFLEN):
                        self.buf = self.buf[1:]

                    # passing judgement lch
                    if wnum == 0:
                        zero_count += 1
                        if non_zero_count > 0:
                            if zero_count < self.zerotorelance:
                                non_zero_count += 1
                            elif zero_count == self.zerotorelance:
                                passtime = non_zero_count - self.zerotorelance
                                wnum_sum = sum(self.buf[-non_zero_count:])
                                if passtime > 10 and wnum_sum > self.area_thresh: #area thresh:5000
                                    # 10,5000 are golden numbers
                                    self.numcount += 1
                                    # Threading background
                                    if self.recthread == None or self.recthread.is_alive() == False:
                                        #self.sendtoslack('A person has just passed by! Time:')

                                        self.recthread = threading.Thread(
                                            target=self.recresult)
                                        self.recthread.start()

                                non_zero_count = 0
                    else:
                        zero_count = 0
                        non_zero_count += 1

                    img_show = np.zeros([h_s, 2*w_s, 3], dtype=np.uint8)
                    text = 'num: ' + str(self.numcount) + \
                        ', diff: '+str(difnum)
                    cv2.putText(mask, text,
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 0), 1, cv2.LINE_AA)

                    img_show[:, :w_s, :] = frame
                    img_show[:, w_s:, 0] = mask
                    img_show[:, int(w_s*1.5), 2] = 255
                    if REC == True:
                        writer.write(img_show)
                    cv2.imshow('test', img_show)
                    key_ = cv2.waitKey(3)
                    if key_ == 27 & 0xff or key_ == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    frame_pre = frame_g
                    framecount += 1
            cv2.destroyAllWindows()
            plt.plot(timeframe, diff_graph)
            plt.show()
            self.cap.release()
            if REC == True:
                writer.release()
        else:
            return

    def get_subscription(self, image1, image2):
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgmask = fgbg.apply(image1)
        fgmask = fgbg.apply(image2)
        return fgmask


if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyzer = Walkpass_Analyzer()
        analyzer.setcap(sys.argv[1])
        analyzer.startcap(False)
