# Copyright 2017 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Main program implementing the deep learning algorithms"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import argparse
from cnn_svm import CNN
from gru_svm import GruSvm
from mlp_svm import MLP
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data import load_data
from utils.data import one_hot_encode

from tkinter import PhotoImage, INSERT, END
import tkinter.font as tkFont
import tkinter.font as font
import pymysql
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import tensorflow as tf



BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_RATE = 0.85
LEARNING_RATE = 1e-3
NODE_SIZE = [512, 256, 128]
NUM_LAYERS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Learning Using Support Vector Machine for Malware Classification"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-m",
        "--model",
        required=True,
        type=int,
        help="[1] CNN-SVM, [2] GRU-SVM, [3] MLP-SVM",
    )
    group.add_argument(
        "-d", "--dataset", required=True, type=str, help="the dataset to be used"
    )
    group.add_argument(
        "-n", "--num_epochs", required=True, type=int, help="number of epochs"
    )
    group.add_argument(
        "-c",
        "--penalty_parameter",
        required=True,
        type=float,
        help="the SVM C penalty parameter",
    )
    group.add_argument(
        "-k",
        "--checkpoint_path",
        required=True,
        type=str,
        help="path where to save the trained model",
    )
    group.add_argument(
        "-l",
        "--log_path",
        required=True,
        type=str,
        help="path where to save the TensorBoard logs",
    )
    group.add_argument(
        "-r",
        "--result_path",
        required=True,
        type=str,
        help="path where to save actual and predicted labels array",
    )
    arguments = parser.parse_args()
    return arguments


def main(model,modelpath,datasetpath):

    model_choice = model
    assert (
        model_choice == 1 or model_choice == 2 or model_choice == 3
    ), "Invalid choice: Choose among 1, 2, and 3 only."

    dataset = np.load(datasetpath,allow_pickle=True)

    features, labels = load_data(dataset=dataset)

    labels = one_hot_encode(labels=labels)

    # get the number of features
    num_features = features.shape[1]

    # get the number of classes
    num_classes = labels.shape[1]

    # split the dataset by 70/30
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.30, stratify=labels
    )

    train_size = int(train_features.shape[0])
    train_features = train_features[: train_size - (train_size % BATCH_SIZE)]
    train_labels = train_labels[: train_size - (train_size % BATCH_SIZE)]

    test_size = int(test_features.shape[0])
    test_features = test_features[: test_size - (test_size % BATCH_SIZE)]
    test_labels = test_labels[: test_size - (test_size % BATCH_SIZE)]
    from pathlib import Path
    cwd=os.getcwd()
    training_dir = Path((os.path.join(cwd,"training")))
    training_dir = str(training_dir)
    print(training_dir)

    log_dir = Path((os.path.join(cwd,"log")))
    log_dir = str(log_dir)

    result_dir = Path((os.path.join(cwd,"result")))
    result_dir = str(result_dir)
    if model_choice == 1:
        model = CNN(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            num_classes=num_classes,
            penalty_parameter=10,
            sequence_length=num_features,
        )
        model.train(
            checkpoint_path= training_dir,
            log_path= log_dir,
            result_path=result_dir,
            epochs=10,
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
        )
    elif model_choice == 2:
        train_features = np.reshape(
            train_features,
            (
                train_features.shape[0],
                int(np.sqrt(train_features.shape[1])),
                int(np.sqrt(train_features.shape[1])),
            ),
        )
        test_features = np.reshape(
            test_features,
            (
                test_features.shape[0],
                int(np.sqrt(test_features.shape[1])),
                int(np.sqrt(test_features.shape[1])),
            ),
        )
        model = GruSvm(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            cell_size=CELL_SIZE,
            dropout_rate=DROPOUT_RATE,
            num_classes=num_classes,
            num_layers=NUM_LAYERS,
            sequence_height=train_features.shape[2],
            sequence_width=train_features.shape[1],
            svm_c=10,
        )
        model.train(
            checkpoint_path='checkpoint/',
            log_path='checkpoint/',
            epochs='checkpoint/',
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
            result_path='checkpoint/',
        )
    elif model_choice == 3:
        model = MLP(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            node_size=NODE_SIZE,
            num_classes=num_classes,
            num_features=num_features,
            penalty_parameter=10,
        )
        model.train(
            checkpoint_path='checkpoint/',
            num_epochs='checkpoint/',
            log_path='checkpoint/',
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
            result_path='checkpoint/',
        )


# if __name__ == "__main__":
#     args = parse_args()

#     main(args)



# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class GUIMAIN:
    folder_selected=''
    def gui(self):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
        root = tk.Tk()
        root.title("Malware Image Classifier")
        canvas = tk.Canvas(root, height="1000", width="1000", bg="white", bd="4")
        canvas.pack()

        myFont = font.Font(family='Helvetica', size="9", weight="bold")
        myFont1 = font.Font(family='Helvetica', size="13", weight="bold")
        myFont2 = font.Font(family='Helvetica', size="15", weight="bold")

        label = tk.Label(canvas, anchor="n", text="Malware image Classifier", width="20", font=("bold", 20))
        label.place(x=170, y=23)

    #In this function we will get path of dataset folder
        def showimage():
            self.folder_selected = filedialog.askopenfilename()
            print(self.folder_selected)
            brows.insert(10,self.folder_selected)
    # fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File",
            #                                      filetypes=(("JPG File", "*.jpg"), ("PNG file", "*.png"), ("ALL Files", "*.*")))
            #     #img = Image.open(fln)
            #     #img = ImageTk.PhotoImage(img)
            #     #lbl.configure(image=img)
            #     #lbl.image = img

        my_tab = ttk.Notebook(canvas)
        my_tab.pack(pady="70")
        frame1 = tk.Frame(my_tab, width="700", height="600", bg="gray")
        frame2 = tk.Frame(my_tab, width="700", height="600", bg="gray")
        frame3 = tk.Frame(my_tab, width="700", height="600", bg="gray")
        frame1.pack(fill="both", expand="0")
        frame2.pack(fill="both", expand="0")
        frame3.pack(fill="both", expand="0")

        my_tab.add(frame1, text="load image")
        my_tab.add(frame2, text="load TextFile")
        my_tab.add(frame3, text="Analyze")


        label_1_1 = tk.Label(frame1, width="20", text="Select Algorithm / model", bg="gray")
        label_1_1['font'] = myFont1
        label_1_1.place(x=80, y=50)

        n = tk.StringVar()
        monthchoosen = ttk.Combobox(frame1, width=27,
                                textvariable=n)

    # Adding combobox drop down list
        monthchoosen['values'] = ('cnn_svm',
                              'gru_svm',
                              'mlp_svm'
                              )

    # Shows february as a default value
    # monthchoosen.current(0)
        monthchoosen.place(x="350", y="50")



        label_1 = tk.Label(frame1, width="20", text="Image DataSet", bg="gray")
        label_1['font'] = myFont1
        label_1.place(x=45, y=120)

        brows = tk.Entry(frame1, width="50", bd="4.5")
        brows.place(x="90", y="145")

        Button_1 = tk.Button(frame1, text="Browse", width="20", bd="3", command=showimage)
        Button_1.place(x="401", y="146")
        Button_1['font'] = myFont

        Button_2 = tk.Button(frame1, text="Load DataSet", width="20", bd="3")
        Button_2.place(x="90", y="190")
        Button_2['font'] = myFont

        def runclassifier():
            val = monthchoosen.get()
            # print(val)
            if(val=='cnn_svm'):
               model=1
               modelpath="E:\Freelancing\shahidWork\Training algos"
               print(model)
               print(modelpath)
            elif(val=='gru_svm'):
               model=2
               print(model)
            elif(val=='mlp_svm'):
               model=3
               print(model)

            #Getting dataset path
            datasetpath=self.folder_selected
            print(datasetpath)
            #Now Passing values to classifier.py file for further evaluation and calculation
            main(model,modelpath,datasetpath)




        Button_3 = tk.Button(frame1, text="Run Classifier", width="20", bd="3", command=runclassifier)
        Button_3.place(x="350", y="190")
        Button_3['font'] = myFont


        label_2 = tk.Label(frame1, width="20", text="Results", bg="gray")
        label_2['font'] = myFont2
        label_2.place(x="15", y="250")

        frm = tk.Canvas(frame1, height="200", width="450", bg="white", bd="3")
        frm.place(x="90",y="280")

            # Result = tk.Entry(frame1, width="50", bd="4")
            # Result.place(x="90", y="300")

        label_3 = tk.Label(frame2, width="20", text="Textual DataSet", bg="gray")
        label_3['font'] = myFont1
        label_3.place(x=45, y=120)

        brows_1 = tk.Entry(frame2, width="50", bd="4.5")
        brows_1.place(x="90", y="145")

        Button_3 = tk.Button(frame2, text="Browse", width="20", bd="3", command=showimage)
        Button_3.place(x="401", y="146")
        Button_3['font'] = myFont

        Button_4 = tk.Button(frame2, text="Load Text DataSet", width="20", bd="3")
        Button_4.place(x="90", y="190")
        Button_4['font'] = myFont

        label_3 = tk.Label(frame2, width="20", text="Results", bg="gray")
        label_3['font'] = myFont2
        label_3.place(x="13", y="250")

        frm = tk.Canvas(frame2, height="200", width="450", bg="white", bd="3")
        frm.place(x="90", y="280")

        Button_5 = tk.Button(frame3, text="Analyze Text DataSet", width="20", bd="3")
        Button_5.place(x="90", y="130")
        Button_5['font'] = myFont

        Button_6 = tk.Button(frame3, text="Analyze Image DataSet", width="20", bd="3")
        Button_6.place(x="350", y="130")
        Button_6['font'] = myFont

        Button_7 = tk.Button(frame3, text="Eval Text Report ", width="20", bd="3")
        Button_7.place(x="90", y="180")
        Button_7['font'] = myFont

        Button_8 = tk.Button(frame3, text="Eval Image Report ", width="20", bd="3")
        Button_8.place(x="350", y="180")
        Button_8['font'] = myFont

        frm1 =tk.Frame(frame3, height="200", width="410", bg="white", bd="3")
        frm1.place(x="90",y="230")


        user = 'root'
        password = 'shuja123@'
        host = 'localhost'
        database = 'Malware'

        # con = pymysql.connect(host="localhost",user= "root",password= "shuja123@", database="Malware")
        con = pymysql.connect(host="localhost",user= "root",password= "shuja123@", database="Malware")

            #prepare a cursor object using cursor() method
        cursor = con.cursor()

            # execute SQL query using execute() method.
        cursor.execute("SELECT VERSION()")

            # Fetch a single row using fetchone() method.
        data = cursor.fetchone()
        print("Database version : %s " % data)

            # disconnect from server
        con.close()

        root.mainloop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   g=GUIMAIN()
   g.gui()


