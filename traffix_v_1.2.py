from customtkinter import *
from tkinter import messagebox,ttk,Canvas
from PIL import Image,ImageTk
from ultralytics import YOLO
from tracker import *
import cv2
import os

class ObjectDetector:
    
    def __init__(self, filename):

        # Initializing the user input video using opencv
        self.file_name = filename
        self.cap = cv2.VideoCapture(filename)
        self.model = YOLO(r"yolov8s.pt")

        # self.tracker makes sure that the detected object isnt detected as a new object again
        self.tracker = EuclideanDistTracker()
        
        # Initializing the counters for RED mode and GREEN mode 
        self.counter_red = 0
        self.counter_green = 0

        # Creating the window that contains the canvas and buttons
        self.root = CTk()
        self.root.iconbitmap(r'trfx.ico')
        self.root.title("Object Detection")
        self.root.resizable(width=False,height=False)
        self.root.protocol("WM_DELETE_WINDOW",self.nextButton)

        # Creating the style object to modify the button apearences
        self.green_button_style = ttk.Style()
        self.green_button_style.configure(
            "Green.TRadiobutton", 
            background = 'GRAY', 
            foreground = 'GREEN')

        self.red_button_style = ttk.Style()
        self.red_button_style.configure(
            "Red.TRadiobutton", 
            background = 'GRAY', 
            foreground = 'RED')
        
        # Creating the canvas that holds the processed frame
        self.canvas = Canvas(self.root, width=640, height=480,highlightthickness=0)
        self.canvas.pack(side=LEFT,padx=10,pady=10)

        # Creating the frame to store the RED, GREEN and NEXT buttons
        self.radio_frame = CTkFrame(self.root)
        self.radio_frame.pack(side=RIGHT, padx=20,ipadx=5,ipady=5)

        # Creating the RED, GREEN and NEXT buttons
        self.radio_var = IntVar()
        self.radio_button1 = CTkRadioButton(self.radio_frame, text="Red", variable=self.radio_var, value=1,fg_color="#C70039",hover_color="#9d002d",text_color="#C70039",font=("Sofachrome Rg",9))
        self.radio_button2 = CTkRadioButton(self.radio_frame, text="Green", variable=self.radio_var, value=0,fg_color="#1D8348",hover_color="#14562f",text_color="#1D8348",font=("Sofachrome Rg",9))
        self.next_button = CTkButton(self.radio_frame, text="Next",command=self.nextButton,corner_radius=100,font=("Sofachrome Rg",7),fg_color="#C70039",hover_color="#9d002d")
        self.radio_button1.pack(anchor=W)
        self.radio_button2.pack(anchor=W)
        self.next_button.pack(anchor=W)

    def detect(self):
        self.id_index = []
        _,self.first_frame = self.cap.read()
        self.frame_width = self.first_frame.shape[1]
        while True:
            detections = []
            self.ret, frame = self.cap.read()
            if self.ret:
                roi = frame[340:720, :]
                results = self.model.predict(roi)
                result = results[0]
                cv2.line(frame, (0, 600), (self.frame_width, 600), color=(0, 255, 0), thickness=2)
                cv2.putText(frame, str(self.counter_red), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, str(self.counter_green), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                for box in result.boxes:
                    conf = round(box.conf.item(), 2)
                    class_id = result.names[box.cls[0].item()]
                    if conf > 0.40 and class_id in ['car', 'truck','bike']:
                        cords = box.xyxy[0].tolist()
                        cords = [round(x) for x in cords]
                        detections.append(cords)

                detections_with_ids = self.tracker.update(detections)

                for detection in detections_with_ids:
                    cx = ((detection[0] + detection[2]) // 2)
                    cy = ((detection[1] + detection[3]) // 2) + 350

                    cv2.circle(frame, (cx, cy), color=(255, 0, 0), thickness=10, radius=5)
                    cv2.rectangle(frame, (detection[0], detection[1] + 350), (detection[2], detection[3] + 350),
                                  color=(0, 0, 0), thickness=2)
                    cv2.putText(frame, str(detection[-1]), (detection[0], detection[1] + 350), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 255), 1)
                    if cy > 600 and detection[-1] not in self.id_index and cy < 610:
                        if self.radio_var.get() == 0:
                            self.counter_green += 1
                        else:
                            self.counter_red += 1
                        self.id_index.append(detection[-1])

                # Convert the OpenCV frame to a PhotoImage object for displaying on the canvas
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640,480))
                img = ImageTk.PhotoImage(master=self.root,image=Image.fromarray(img))
                self.canvas.create_image(0, 0, anchor=NW, image=img)
                #self.canvas.photo = img
                self.root.update()

            else:
                self.cap.release()
                cv2.destroyAllWindows()
                self.root.destroy()
                FinalWindow(self.counter_red,(self.counter_red+self.counter_green),self.file_name)
    
    def nextButton(self):
        self.cap.release()

class MainWindow:

    def __init__(self):
        self.root = CTk()
        
        # Setting the icon of the app
        self.root.iconbitmap(r'trfx.ico')
        
        # Geometry of the window
        self.width = 1280
        self.height = 720
        self.root.resizable(width=False, height=False)

        # Getting the width and height 
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Centering the window on the users display
        self.x = (self.screen_width/2) - (self.width/2)
        self.y = (self.screen_height/2) - (self.height/2)
        self.root.geometry('%dx%d+%d+%d' % (self.width, self.height, self.x, self.y))
        self.root.grid_columnconfigure(0,weight=1)
        self.root.geometry('%dx%d+%d+%d' % (self.width, self.height, self.x, self.y))
        
        # Widgets used in the main window
        self.frame = CTkFrame(self.root,corner_radius=17)
        self.spacer = CTkLabel(self.root,text="")
        self.title = CTkLabel(self.root,text='Traffix™',font=("Sofachrome Rg",90))
        self.slogan = CTkLabel(self.root,text='Red Light. Brake Right.',font=("Sofachrome Rg",20),fg_color="#C70039",corner_radius=17)
        self.browse_button = CTkButton(self.frame,text="Browse",corner_radius=1000,command=self.browseFile,font=("Sofachrome Rg",9),fg_color="#CEAC35",hover_color="#a4892e")
        self.search_box = CTkEntry(self.frame,placeholder_text="Paste the file path here or press the browse button...",width=700,corner_radius=1000,font=("Sofachrome Rg",10))
        self.exit_button = CTkButton(self.frame,text="Exit",corner_radius=1000,command=self.root.destroy,font=("Sofachrome Rg",9),fg_color="#1D8348",hover_color="#14562f")
        self.open_button = CTkButton(self.frame,text="Open",corner_radius=1000,command=self.openFile,font=("Sofachrome Rg",9),fg_color="#C70039",hover_color="#9d002d")

        # Widget positioning
        self.frame.grid(row=3,column=0,columnspan=3,ipadx=5,ipady=5,padx=8,sticky="N")
        self.title.grid(row=0,pady=10)
        self.slogan.grid(row=1,pady=10)
        self.spacer.grid(row=2,pady=100)
        self.open_button.grid(row=0,column=1,pady=5)
        self.search_box.grid(row=0,column=0,padx=10,pady=5)
        self.browse_button.grid(row=1,column=1)
        self.exit_button.grid(row=2,column=1,pady=5)

        # Running the window
        self.root.mainloop()


    # Function to open given file
    def openFile(self):
        
        self.file_path = self.search_box.get()
        if os.path.exists(self.file_path):
            self.root.withdraw()
            obj = ObjectDetector(self.file_path)
            obj.detect()
        else:
            messagebox.showerror(title="Error",message="File Not Found")

    # Function to browse for a file and to open it
    def browseFile(self):
        
        self.file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if os.path.exists(self.file_path):
            self.root.withdraw()
            obj = ObjectDetector(self.file_path)
            obj.detect()
        else:
            messagebox.showerror(title="Error",message="File Not Found")

class FinalWindow:

    def __init__(self,counter_red,counter_green,filename):

        # Getting the counter values as parameters
        self.file_name = filename
        self.counter_red = counter_red
        self.counter_green = counter_green

        # Creating the window
        self.root = CTk()
        
        # Setting the icon of the app
        self.root.iconbitmap(r'trfx.ico')
        
        # Geometry of the window
        self.width = 1280
        self.height = 720
        self.root.resizable(width=False, height=False)

        # Getting the width and height 
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Centering the window on the users display
        self.x = (self.screen_width/2) - (self.width/2)
        self.y = (self.screen_height/2) - (self.height/2)
        self.root.geometry('%dx%d+%d+%d' % (self.width, self.height, self.x, self.y))
    
        # Widgets
        self.frame = CTkFrame(self.root)
        self.frame_ = CTkFrame(self.frame,corner_radius=25)
        self.title = CTkLabel(self.frame,text="Traffix™",font=("Sofachrome Rg",70))
        self.spacer = CTkLabel(self.frame,text="")
        self.file_name_box = CTkLabel(self.frame_,text=self.file_name.split("/")[-1],font=("Sofachrome Rg",30),fg_color="#C70039",corner_radius=100)
        self.total_vehicles = CTkLabel(self.frame_,text="TOTAL VEHICLES:  "+str(self.counter_green+self.counter_red),font=("Sofachrome Rg",15),text_color="#1D8348")
        self.violations = CTkLabel(self.frame_,text="NO. OF VIOLATIONS:  "+str(self.counter_red),font=("Sofachrome Rg",15),text_color="#C70039")
        self.return_button = CTkButton(self.root,text="Return",hover=True,corner_radius=1000,font=("Sofachrome Rg",9),fg_color="#C70039",hover_color="#9d002d",command=self.toMainWindow)
        
        # Widget Positioning
        self.frame.pack(fill=BOTH,expand=True,pady=45,padx=20)
        self.title.pack(anchor=CENTER,padx=10,pady=5)
        self.spacer.pack(pady=100)
        self.frame_.pack(ipadx=100,ipady=10)
        self.file_name_box.pack(anchor=CENTER,padx=10,pady=10)
        self.total_vehicles.pack(anchor=W,padx=10,pady=5)
        self.violations.pack(anchor=W,padx=10,pady=5)
        self.return_button.place(relx=0.885,rely=0.95)
        
        self.root.protocol("WM_DELETE_WINDOW", MainWindow)
        self.root.mainloop()
        
    def toMainWindow(self):

        self.root.destroy()
        MainWindow()


MainWindow()