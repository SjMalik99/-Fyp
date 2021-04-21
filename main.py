# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import PhotoImage, INSERT, END
import tkinter.font as tkFont
import tkinter.font as font
import pymysql
import cv2
import tensorflow as tf




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import os

    root = tk.Tk()
    root.title("Malware Image Classifier")
    canvas = tk.Canvas(root, height="1000", width="1000", bg="white", bd="4")
    canvas.pack()

    myFont = font.Font(family='Helvetica', size="9", weight="bold")
    myFont1 = font.Font(family='Helvetica', size="13", weight="bold")
    myFont2 = font.Font(family='Helvetica', size="15", weight="bold")

    label = tk.Label(canvas, anchor="n", text="Malware image Classifier", width="20", font=("bold", 20))
    label.place(x=170, y=23)

    def filedialog():
        filename = filedialog.askopenfilename(  initialdir="/", tile="Select A File", filetype=(("jpeg", "*.jpg"),
                                                                                              ("All Files", "*.*")))

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

    label_1 = tk.Label(frame1, width="20", text="Image DataSet", bg="gray")
    label_1['font'] = myFont1
    label_1.place(x=45, y=120)

    brows = tk.Entry(frame1, width="50", bd="4.5")
    brows.place(x="90", y="145")

    Button_1 = tk.Button(frame1, text="Browse", width="20", bd="3", command=filedialog)
    Button_1.place(x="401", y="146")
    Button_1['font'] = myFont

    Button_2 = tk.Button(frame1, text="Load DataSet", width="20", bd="3")
    Button_2.place(x="90", y="190")
    Button_2['font'] = myFont

    label_2 = tk.Label(frame1, width="20", text="Results", bg="gray")
    label_2['font'] = myFont2
    label_2.place(x="15", y="250")

    frm = tk.Canvas(frame1, height="200", width="450", bg="white", bd="3")
    frm.place(x="90", y="280")

    # Result = tk.Entry(frame1, width="50", bd="4")
    # Result.place(x="90", y="300")

    label_3 = tk.Label(frame2, width="20", text="Textual DataSet", bg="gray")
    label_3['font'] = myFont1
    label_3.place(x=45, y=120)

    brows_1 = tk.Entry(frame2, width="50", bd="4.5")
    brows_1.place(x="90", y="145")

    Button_3 = tk.Button(frame2, text="Browse", width="20", bd="3")
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
    password = ''
    host = 'localhost'
    database = 'Malware_img'

    con = pymysql.connect(host='localhost',user= 'root', password= 'Shahid123@', database= 'Malware_img')

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
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
