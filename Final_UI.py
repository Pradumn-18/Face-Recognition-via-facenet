from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import PIL
import xyz_rc
import sqlite3
from classifier import training
import pickle
from preprocess import preprocesses
import subprocess
from tkinter import *
import tkinter.messagebox
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(631, 443)
        Dialog.setStyleSheet("background-image: url(:/newPrefix/background.jpg);")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(210, 380, 391, 41))
        self.buttonBox.setStyleSheet("color: rgb(255, 255, 255);")
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(440, 210, 151, 41))
        self.pushButton.setStyleSheet("color: rgb(255, 255, 255);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(440, 70, 151, 41))
        self.pushButton_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(440, 280, 151, 41))
        self.pushButton_3.setStyleSheet("color: rgb(255, 255, 255);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(440, 140, 151, 41))
        self.pushButton_4.setStyleSheet("color: rgb(255, 255, 255);")
        self.pushButton_4.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 30, 351, 371))
        self.label.setStyleSheet("selection-background-color: rgb(255, 255, 255);")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("p1.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.pushButton.clicked.connect(self.identify_face_video)
        self.pushButton_3.clicked.connect(self.database_fetch)
        self.pushButton_2.clicked.connect(self.form)
        self.pushButton_4.clicked.connect(self.train)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Recognise"))
        self.pushButton_2.setText(_translate("Dialog", "Add a Face"))
        self.pushButton_3.setText(_translate("Dialog", "Fetch Details "))
        self.pushButton_4.setText(_translate("Dialog", "Train"))
        
        
    
        
    
    def identify_face_video(self):
        modeldir = './model/20170511-185253.pb'
        classifier_filename = './class/classifier.pkl'
        npy='./npy'
        train_img="./train_img"
        
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        
                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                margin = 44
                frame_interval = 3
                batch_size = 1000
                image_size = 182
                input_image_size = 160
                
                HumanNames = os.listdir(train_img)
                HumanNames.sort()
        
                print('Loading Modal')
                facenet.load_model(modeldir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
        
        
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
        
                video_capture = cv2.VideoCapture(0)
                c = 0
        
        
                print('Start Recognition')
                prevTime = 0
                while True:
                    ret, frame = video_capture.read()
        
                    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
        
                    curTime = time.time()+1    # calc fps
                    timeF = frame_interval
        
                    if (c % timeF == 0):
                        find_results = []
        
                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('Detected_FaceNum: %d' % nrof_faces)
        
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(frame.shape)[0:2]
        
                            cropped = []
                            scaled = []
                            scaled_reshape = []
                            bb = np.zeros((nrof_faces,4), dtype=np.int32)
                            
                            try:
                                for i in range(nrof_faces):
                                    emb_array = np.zeros((1, embedding_size))
                                    
                                    bb[i][0] = det[i][0]
                                    bb[i][1] = det[i][1]
                                    bb[i][2] = det[i][2]
                                    bb[i][3] = det[i][3]
                                    
                                    # inner exception
                                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                        print('Face is very close!')
                                        continue
                                    
                                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                    cropped[i] = facenet.flip(cropped[i], False)
                                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                           interpolation=cv2.INTER_CUBIC)
                                    scaled[i] = facenet.prewhiten(scaled[i])
                                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                    predictions = model.predict_proba(emb_array)
                                    print(predictions)
                                    best_class_indices = np.argmax(predictions, axis=1)
                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                    # print("predictions")
                                    print(best_class_indices,' with accuracy ',best_class_probabilities)
                                    
                                    # print(best_class_probabilities)
                                    if best_class_probabilities>0.85:
                                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                                        
                                        #plot result idx under box
                                        text_x = bb[i][0]
                                        text_y = bb[i][3] + 20
                                        print('Result Indices: ', best_class_indices[0])
                                        print(HumanNames)
                                        global getName
                                        getName = best_class_indices[0]
                                        global name,fetch
                                        name=HumanNames[getName]
                                        for H_i in HumanNames:
                                            if HumanNames[best_class_indices[0]] == H_i:
                                                result_names = HumanNames[best_class_indices[0]]
                                                predict_name = result_names[ :-17]
                                                fetch = result_names[-16:-1]
                                                cv2.putText(frame, predict_name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                            1, (0, 0, 255), thickness=1, lineType=2)
                            except IndexError:
                                print("Oops! IndexError : list index out of range for multi_faces")
                        else:
                            print('Alignment Failure')
                    # c+=1
                    cv2.imshow('Video', frame)
        
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
                video_capture.release()
                cv2.destroyAllWindows()

    def train(self):
        #data_preprocess
        root1 = Tk()
        root1.geometry('400x400')
        input_datadir = './train_img'
        output_datadir = './pre_img'
        
        obj=preprocesses(input_datadir,output_datadir)
        nrof_images_total,nrof_successfully_aligned=obj.collect_data()
        
        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
        
        w = Label(root1,text=str(nrof_images_total))
        w1 = Label(root1,text=str(nrof_successfully_aligned))
        
        w.pack()
        w1.pack()
        
        
        #For Training
        root = Tk()
        root.geometry('400x400')
        root.title('Training Status')
        datadir = './pre_img'
        modeldir = './model/20170511-185253.pb'
        classifier_filename = './class/classifier.pkl'
        print ("Training Start")
        obj=training(datadir,modeldir,classifier_filename)
        get_file=obj.main_train()
        print('Saved classifier model to file "%s"' % get_file)
        
        ourMessage = "All Done"
        p = Label(root,text=ourMessage)
        p.place(x=45,y=80)
        root.mainloop()
        #sys.exit("All Done")
        
        
        
    
    def database_fetch(self):
        root =Tk()
        root.geometry('400x600')
        root.title('Details')
        conn = sqlite3.connect('Employee_Details.db')
        sql = "SELECT eid,unique_id,name_of_company,f_name,s_name,dob,father_name,mother_name,designation,gender,issue_date,valid_upto,address,pincode,phone_no,email FROM EMPLOYEE_DATA WHERE eid = :id"            
        param = {'id' : fetch}
        cursor = conn.execute(sql,param)
        for row in cursor:
            ourMessage = str(row[0])
            ourMessage1 = str(row[1])
            ourMessage2 = str(row[2])
            ourMessage3 = str(row[3])
            ourMessage4 = str(row[4])
            ourMessage5 = str(row[5])
            ourMessage6 = str(row[6])
            ourMessage7 = str(row[7])
            ourMessage8 = str(row[8])
            ourMessage9 = str(row[9])
            ourMessage10 = str(row[10])
            ourMessage11 = str(row[11])
            ourMessage12 = str(row[12])
            ourMessage13 = str(row[13])
            ourMessage14 = str(row[14])
            ourMessage15 = str(row[15])
        
        w = Label(root,text='Employee ID = '+ourMessage,font=("bold",12))
        w1 = Label(root,text='ID No.= '+ourMessage1,font=("bold",12))
        w2 = Label(root,text='Name of Company = '+ourMessage2,font=("bold",12))
        w3 = Label(root,text='First Name = '+ourMessage3,font=("bold",12))
        w4 = Label(root,text='Surname = '+ourMessage4,font=("bold",12))
        w5 = Label(root,text='Date Of Birth = '+ourMessage5,font=("bold",12))
        w6 = Label(root,text='Father Name = '+ourMessage6,font=("bold",12))
        w7 = Label(root,text='Mother Name = '+ourMessage7,font=("bold",12))
        w8 = Label(root,text='Designation = '+ourMessage8,font=("bold",12))
        w9 = Label(root,text='Gender = '+ourMessage9,font=("bold",12))
        w10 = Label(root,text='Issue_Date = '+ourMessage10,font=("bold",12))
        w11 = Label(root,text='Valid_Upto = '+ourMessage11,font=("bold",12))
        w12 = Label(root,text='Address = '+ourMessage12,font=("bold",12))
        w13 = Label(root,text='Pincode = '+ourMessage13,font=("bold",12))
        w14 = Label(root,text='Contact = '+ourMessage14,font=("bold",12))
        w15 = Label(root,text='Email = '+ourMessage15,font=("bold",12))
        w.config(bg='yellow')
        w1.config(bg='yellow')
        w2.config(bg='yellow')
        w3.config(bg='yellow')
        w4.config(bg='lightgreen')
        w5.config(bg='lightgreen')
        w6.config(bg='lightgreen')
        w7.config(bg='lightgreen')
        w8.config(bg='yellow')
        w9.config(bg='lightgreen')
        w10.config(bg='yellow')
        w11.config(bg='yellow')
        w12.config(bg='lightgreen')
        w13.config(bg='lightgreen')
        w14.config(bg='lightgreen')
        w15.config(bg='yellow')
        
        w.place(x=85,y=80)
        w1.place(x=85,y=110)
        w2.place(x=85,y=140)
        w3.place(x=85,y=170)
        w4.place(x=85,y=200)
        w5.place(x=85,y=230)
        w6.place(x=85,y=260)
        w7.place(x=85,y=290)
        w8.place(x=85,y=320)
        w9.place(x=85,y=350)
        w10.place(x=85,y=380)
        w11.place(x=85,y=410)
        w12.place(x=85,y=440)
        w13.place(x=85,y=470)
        w14.place(x=85,y=500)
        w15.place(x=85,y=530)
        
        root.mainloop()
    
    def form(self):
        root=Tk()
        root.geometry('950x900')
        root.title("registration form")
        
        eid=IntVar()
        unique_id=StringVar()
        #name_of_company=StringVar()
        f_name=StringVar()
        s_name=StringVar()
        dob=StringVar()
        father_name=StringVar()
        mother_name=StringVar()
        designation=StringVar()
        gender=StringVar()
        issue_date=StringVar()
        valid_upto=StringVar()
        address=StringVar()
        pincode=IntVar()
        phone_no=IntVar()
        email=StringVar()
        c=StringVar()
        g=StringVar()
        
        def Add_Image():
            subprocess.Popen('explorer "C:\\Users\\Admin\\Downloads\\Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow-master\\Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow-master\\train_img" ')
            
        def database():
            #eid_=eid.get()
            unique_id_=unique_id.get()
            name_of_company_=g.get()
            f_name_=f_name.get()
            s_name_=s_name.get()
            dob_=dob.get()
            father_name_=father_name.get()
            mother_name_=mother_name.get()
            designation_=designation.get()
            gender_=gender.get()
            issue_date_=issue_date.get()
            valid_upto_=valid_upto.get()
            address_=address.get()
            pincode_=pincode.get()
            phone_no_=phone_no.get()
            email_=email.get()
            id_type_=c.get()
            conn = sqlite3.connect('Employee_Details.db')
            with conn:
                cursor=conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS EMPLOYEE_DATA (eid CHAR(15) PRIMARY KEY NOT NULL,unique_id TEXT UNIQUE,id_type Text ,name_of_company TEXT , f_name TEXT NOT NULL, s_name TEXT, dob TEXT CHECK(LENGTH(dob) == 10) NOT NULL, father_name TEXT NOT NULL, mother_name TEXT NOT NULL, designation TEXT NOT NULL, gender TEXT NOT NULL, issue_date TEXT CHECK(LENGTH(dob) == 10) NOT NULL, valid_upto TEXT CHECK(LENGTH(dob) == 10)  NOT NULL, address TEXT NOT NULL, pincode INT NOT NULL, phone_no INT NOT NULL,email TEXT NOT NULL)')
            cursor.execute('INSERT INTO EMPLOYEE_DATA (eid,unique_id,name_of_company,f_name,s_name,dob,father_name,mother_name,designation,gender,issue_date,valid_upto,address,pincode,phone_no,email) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(EID,unique_id_,name_of_company_,f_name_,s_name_,dob_,father_name_,mother_name_,designation_,gender_,issue_date_,valid_upto_,address_,pincode_,phone_no_,email_))
            conn.commit()
            root1=Tk()
            root1.geometry('300x400')
            root1.title("Pop UP")
            m = "DATA SAVED"
            w = Label(root1,text=m)
            w.place(x=45,y=80)
            root.destroy()
            root1.mainloop()
            
            conn.close()
            
        def Generate_EID():
            unique_id_=unique_id.get()
            name_of_company_=g.get()
            f_name_=f_name.get()
            s_name_=s_name.get()
            dob_=dob.get()
            father_name_=father_name.get()
            mother_name_=mother_name.get()
            designation_=designation.get()
            gender_=gender.get()
            issue_date_=issue_date.get()
            valid_upto_=valid_upto.get()
            address_=address.get()
            pincode_=pincode.get()
            phone_no_=phone_no.get()
            email_=email.get()
            id_type_=c.get()
            #variable for slicing values
            noc_s1=name_of_company_[-3 : ]
            uid_s2=unique_id_[0:3]
            dob_s3=dob_[0:2]
            noc_s4=name_of_company_[0:1]
            d_s5=designation_[0:1]
            #connecting to database
            conn = sqlite3.connect('Employee_Details.db')
            cursor=conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS EMPLOYEE_DATA (eid CHAR(15) PRIMARY KEY NOT NULL,unique_id TEXT UNIQUE,id_type Text,name_of_company TEXT, f_name TEXT NOT NULL, s_name TEXT, dob TEXT CHECK(LENGTH(dob) == 10) NOT NULL, father_name TEXT NOT NULL, mother_name TEXT NOT NULL, designation TEXT NOT NULL, gender TEXT NOT NULL, issue_date TEXT CHECK(LENGTH(dob) == 10) NOT NULL, valid_upto TEXT CHECK(LENGTH(dob) == 10)  NOT NULL, address TEXT NOT NULL, pincode INT NOT NULL, phone_no INT NOT NULL,email TEXT NOT NULL)')
            cursor.execute('SELECT COUNT(*) FROM EMPLOYEE_DATA')
            for row in cursor:
                employee_no=row[0]
            
            employee_no+=1
            if employee_no>0 and employee_no<10:
                str1="0000"+str(employee_no)
            elif employee_no>9 and employee_no<100:
                str1="000"+str(employee_no)
            elif employee_no>99 and employee_no<1000:
                str1="00"+str(employee_no)
            elif employee_no>999 and employee_no<10000:
                str1="0"+str(employee_no)
            elif employee_no>9999 and employee_no<100000:
                str1=str(employee_no)
                
            global EID
            EID = noc_s1+str1+uid_s2+dob_s3+noc_s4+d_s5
            
            root2= Tk()
            root2.geometry('300x300')
            root2.title("Employee ID ")
            n = Label(root2,text=EID,font=("bold",15))
            n.place(x=85,y=80)
            n.config(bg='yellow')
            root2.mainloop()
            conn.close()

        
        label_0 = Label(root, text="Registration Form",width=20,font=("bold", 20))

        label_0.place(x=300,y=53)
            
            
        label_1 = Label(root, text="ID NO.*",width=20,font=("bold", 10))
        
        label_1.place(x=80+240+15+100+20,y=170)
        
        list1 = ['Aadhar','Pan Card','DL','Voter ID'];
        droplist=OptionMenu(root,c,*list1)
        droplist.config(width=20)
        c.set('Select ID Type')
        droplist.place(x=240+80+15,y=170)
        
        
        '''entry_1 = Entry(root,textvar=eid)
        
        entry_1.place(x=240,y=130)'''   
        
        
        label_2 = Label(root, text="ID TYPE",font=("bold", 10))
        
        label_2.place(x=120,y=170)
        
        
        
        entry_2 = Entry(root,textvar=unique_id)
        
        entry_2.place(x=240+80+15+100+150,y=170)
        
        
        
        label_3 = Label(root, text="NAME OF COMPANY*",font=("bold", 10))
        
        label_3.place(x=120,y=210)
        
        list2 = ['ABC-001','PQR-002','XYZ-003'];
        droplist1 = OptionMenu(root,g,*list2)
        droplist1.config(width=20)
        g.set('Select compnay')
        droplist1.place(x=240+80+15,y=210)
        
        
        '''entry_3 = Entry(root,textvar=name_of_company)
        
        
        entry_3.place(x=240+80+15,y=210)'''
        
        
        
        label_4 = Label(root, text="FIRST NAME*",font=("bold", 10))
        
        label_4.place(x=120,y=250)
        
        
        
        entry_4 = Entry(root,textvar=f_name)
        
        entry_4.place(x=240+80+15,y=250)
        
        
        
        label_5 = Label(root, text="SURNAME",font=("bold", 10))
        
        label_5.place(x=120,y=290)
        
        
        
        entry_5 = Entry(root,textvar=s_name)
        
        entry_5.place(x=240+80+15,y=290)
        
        
        
        label_6 = Label(root, text="D.O.B (dd/mm/yyyy)",font=("bold", 10))
        
        label_6.place(x=120,y=330)
        
        
        
        entry_6 = Entry(root,textvar=dob)
        
        entry_6.place(x=240+80+15,y=330)
        
        
        
        label_7 = Label(root, text="FATHER NAME",font=("bold", 10))
        
        label_7.place(x=120,y=370)
        
        
        
        entry_7 = Entry(root,textvar=father_name)
        
        entry_7.place(x=240+80+15,y=370)
        
        
        
        label_8 = Label(root, text="MOTHER NAME",font=("bold", 10))
        
        label_8.place(x=120,y=410)
        
        
        
        entry_8 = Entry(root,textvar=mother_name)
        
        entry_8.place(x=240+80+15,y=410)
        
        
        
        label_9 = Label(root, text="DESIGNATION*",font=("bold", 10))
        
        label_9.place(x=120,y=450)
        
        
        
        entry_9 = Entry(root,textvar=designation)
        
        entry_9.place(x=240+80+15,y=450)
        
        
        
        label_10 = Label(root, text="GENDER",font=("bold",10))
        
        label_10.place(x=120,y=490)
        
        
        
        Radiobutton(root, text="Male",padx = 5,variable=gender, value="Male").place(x=240+80+15,y=490)
        
        Radiobutton(root, text="Female",padx = 20,variable=gender, value="Female").place(x=420+15,y=490)
        
        
        
        label_11 = Label(root, text="ISSUE DATE (dd/mm/yyyy)",font=("bold", 10))
        
        label_11.place(x=120,y=530)
        
        
        
        entry_11= Entry(root,textvar=issue_date)
        
        entry_11.place(x=240+80+15,y=530)
        
        
        
        label_12= Label(root, text="VALID UPTO* (dd/mm/yyyy)",font=("bold", 10))
        
        label_12.place(x=120,y=570)
        
        
        
        entry_12= Entry(root,textvar=valid_upto)
        
        entry_12.place(x=240+80+15,y=570)
        
        
        
        label_13= Label(root, text="ADDRESS",font=("bold", 10))
        
        label_13.place(x=120,y=610)
        
        
        
        entry_13= Entry(root,textvar=address)
        
        entry_13.place(x=240+80+15,y=610)
        
        
        
        label_14 = Label(root, text="PINCODE",font=("bold", 10))
        
        label_14.place(x=120,y=650)
        
        
        
        entry_14= Entry(root,textvar=pincode)
        
        entry_14.place(x=240+80+15,y=650)
        
        
        
        label_15 = Label(root, text="PHONE NO*",font=("bold", 10))
        
        label_15.place(x=120,y=690)
        
        
        
        entry_15= Entry(root,textvar=phone_no)
        
        entry_15.place(x=240+80+15,y=690)
        
        
        
        label_16 = Label(root, text="EMAIL",font=("bold", 10))
        
        label_16.place(x=120,y=730)
        
        
        
        entry_16= Entry(root,textvar=email)
        
        entry_16.place(x=240+80+15,y=730)
        
        
        Button(root, text='Image', width=20,bg='brown',fg='white',command=Add_Image).place(x=280+30+5,y=770)
        
        
        Button(root, text='Save', width=20,bg='orange',fg='white',command=database).place(x=280+30+5+200,y=770)
        
        
        Button(root, text='cancel', width=20,bg='red',fg='white',command=root.destroy).place(x=280+30+5+400,y=770)
        
        label_17= Label(root, text="Note* :- Store images in Folder with Folder Name as : Name(EID) ",width=60, font=("bold",8))
        
        label_17.place(x=240+80+15+20, y=810)
        
        Button(root, text='Generate EID', width=20, bg='green',fg='white',command = Generate_EID).place(x=80+30+5,y=770)
        
        
        
        root.mainloop()
        
        
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

