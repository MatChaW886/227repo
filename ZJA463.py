# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf
from pyb import UART
from pyb import Pin
from pyb import LED

uart = UART(3, 115200)
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
#sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=4000)          # Let the camera adjust.
net = "trained.tflite"
labels = [line.rstrip('\n') for line in open("labels.txt")]



#flag   0读图->1取药->2十字->3发送S，识别数字->找到4 没找到回到2
#
flag=0;
left_finish=0
right_finish=0
number_in=0
numberl_get=[]
numberr_get=[]
clock = time.clock()

#def input_number():
    #appear_times = {}
    #for label in numberl_get:
        #if label in appear_times:
          #appear_times[label] += 1
        #else:
          #appear_times[label] = 1
    #return(max(appear_times, key=lambda x: appear_times[x]))


def output_number(dir,t):
    global flag,left_finish,right_finish,number_in
    if(t==1):
        t=7
    if(dir==0):
        if(t==number_in):
            print(t)
            print('L')
            uart.write('L')
            led = LED(1)
            led.on()#亮
            flag=2
        else:
            if(right_finish==1):
                uart.write('G')
                led = LED(2)
                led.on()#亮
                print("no answer")
                flag=2
            else:
                left_finish=1
    if(dir==1):
        if(t==number_in):
            print(t)
            print('R')
            uart.write('R')
            led = LED(3)
            led.on()#亮
            flag=2
        else:
            if(left_finish==1):
                uart.write('G')
                led = LED(2)
                led.on()#亮
                print("no answer")
                flag=2
            else:
                right_finish=1




while(True):
    clock.tick()
    img = sensor.snapshot()
    if(flag!=2):
        if(uart.any()):
            text=uart.readline()
            if('P') in text:
                #uart.write('')
                left_finish=0
                right_finish=0
                #time.sleep_ms()
                print('get cross')
                print("start search")
                flag=3
                break
    if(flag==1):
        for r in img.find_rects(roi=(0,0,320,240),threshold = 50000):             # search the rectangle
            #img.draw_rectangle(r.rect(), color = (255, 255, 255))   # draw the rectangle
            imgtemp = img.copy(r.rect())
            for obj in tf.classify(net, imgtemp, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5): #TFliteNN
                #if(flag!=1):
                    #break
                predictions_list = list(zip(labels, obj.output()))
                #print(predictions_list)
                temp=0
                for i in range(len(predictions_list)):
                    if (predictions_list[i][1]) > temp:
                        temp = predictions_list[i][1]
                        t=i+1
                        #img.draw_string(r.x(),r.y()+r.h(),str(t),scale=4)
                #print(t)
                if(t==7):
                    if(predictions_list[0][1]>0.15) :
                        t=1
                #uart.write(str(t))
                #time.sleep(1000)
                print("Start")
                print(t)
                if(t==1):
                    uart.write('L')
                    led = LED(1)
                    led.on()#亮
                    flag=4
                    #flag=2
                elif(t==2):
                    uart.write('R')
                    led = LED(2)
                    led.on()#亮
                    flag=4
                    #flag=2
                else:
                    uart.write('G')
                    number_in=t
                    led = LED(3)
                    led.on()#亮
                    flag=2

    if(flag==2):
        print("Wait cross")
        if(uart.any()):
            text=uart.readline()
            if('P') in text:
                #uart.write('')
                left_finish=0
                right_finish=0
                #time.sleep_ms()
                print('get cross')
                print("start search")
                flag=3


    if(flag==3):
        print('searching')
        #print(number_in)
        #p_out.low()
        if(left_finish==0):
            for r in img.find_rects(roi=(0,0,160,240),threshold = 40000):             # 在图像中搜索矩形
                img.draw_rectangle(r.rect(), color = (255, 255, 255))   # 绘制矩形外框，便于在IDE上查看识别到的矩形位置
                imgtemp = img.copy(r.rect())
                for obj in tf.classify(net, imgtemp, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                    predictions_list = list(zip(labels, obj.output()))
                    temp=0
                    for i in range(len(predictions_list)):
                        if (predictions_list[i][1]) > temp:
                            temp = predictions_list[i][1]
                            t=i+1
                            #img.draw_string(r.x(),r.y()+r.h(),str(t),scale=4)
                    output_number(0,t)
                    #print("left"+str(t))

        if(right_finish==0):
            for r in img.find_rects(roi=(160,0,160,240),threshold = 40000):             # 在图像中搜索矩形
                img.draw_rectangle(r.rect(), color = (0, 0, 0))   # 绘制矩形外框，便于在IDE上查看识别到的矩形位置
                imgtemp = img.copy(r.rect())
                for obj in tf.classify(net, imgtemp, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                    predictions_list = list(zip(labels, obj.output()))
                    temp=0
                    for i in range(len(predictions_list)):
                        if (predictions_list[i][1]) > temp:
                            temp = predictions_list[i][1]
                            t=i+1
                            #img.draw_string(r.x(),r.y()+r.h(),str(t),scale=4)
                    output_number(1,t)
                    #print("right"+str(t))



    #print(flag)
    #print(frame)
    #print(clock.fps(), "fps")
