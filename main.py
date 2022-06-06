import cv2
import tensorflow as tf
import numpy as np
from PIL import ImageTk, Image
from object_detection import ObjectDetection
import tkinter as tk
from tkinter import filedialog
import easyocr

MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""
    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()

        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:, :, (2, 1, 0)] # RGB -> BGR
        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
        return outputs[0]

def openfile():
    return configImg(filedialog.askopenfilename())
def main():
    global reader, window, od_model, plate_list_img, plate_list_img_label, plate_list_number_label, label_img
    reader = easyocr.Reader(['en']) 
    # 建立主視窗 Frame
    window = tk.Tk()

    # 設定視窗標題
    window.title('Hello World')

    # 設定視窗大小為 300x100，視窗（左上角）在螢幕上的座標位置為 (250, 150)
    window.geometry("960x480+250+150")
    
    # Load a TensorFlow model:
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())


    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFObjectDetection(graph_def, labels)

    plate_list_img = dict()
    plate_list_img_label = dict()
    plate_list_number_label = dict()


    button = tk.Button(window, text = "Open", command = openfile)
    button.grid(column = 0, row = 0)
    filename = "IMG_2967.jpg"
    # global reader, window, od_model, plate_list_img, plate_list_img_label, plate_list_number_label, label_img
    image = Image.open(filename)
    predictions = od_model.predict_image(image)
    img = cv2.imread(filename)
    
    plate_list = []
    h, w, c = img.shape
    for ret in predictions:
        prob = ret['probability']
        if prob < 0.5: continue
        tagID = ret['tagId']
        tagName = ret['tagName']
        bbox = ret['boundingBox']
        left = bbox['left']
        top = bbox['top']
        width = bbox['width']
        height = bbox['height']
        x1 = int(left*w)
        y1 = int(top*h)
        x2 = x1 + int(width*w)
        y2 = y1 + int(height*h)
        p0 = (max(x1, 15), max(y1-5, 15))
        print(f"probability is {prob}, tag id is {tagID}, tag name is {tagName}")
        print(f"bounding box ({x1}, {x2}, {y1}, {y2})")
        info = "{:.2f}:-{}".format(prob, tagName)
        cv2.rectangle(img, (x1, y1) , (x2, y2), (0, 0, 0), 10)
        cv2.rectangle(img, (x1, y1) , (x1+400, y1-50), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, info, p0, cv2.FONT_ITALIC, 2, (0, 255, 0), 3)
        img_copy = img.copy()
        plate = img_copy[y1:y2, x1:x2]
        plate_list.append(plate)

    #Rearrange colors
    blue,green,red = cv2.split(img)
    img = cv2.merge((red,green,blue))
    img = Image.fromarray(img)
    
    img = ImageTk.PhotoImage(img.resize((640, 360)))
    label_img = tk.Label(window, image=img, bg = "blue")
    label_img.grid(row = 1, column = 0, rowspan = 5, sticky=tk.N)
    for index, img_plate in enumerate(plate_list):
        result = reader.readtext(img_plate, detail = 0)
        plate_list_number_label[f'n{index}'] = tk.Label(window, text=result, bg = "red")
        plate_list_number_label[f'n{index}'].grid(row = 1+index, column = 2, sticky=tk.N)
        #Rearrange colors
        blue,green,red = cv2.split(img_plate)
        img_plate = cv2.merge((red,green,blue))
        img_plate = Image.fromarray(img_plate)
        plate_list_img[f'l{index}'] = ImageTk.PhotoImage(img_plate.resize((160, 90)))
        plate_list_img_label[f'i{index}'] = tk.Label(window, image=plate_list_img[f'l{index}'], bg = "red")
        plate_list_img_label[f'i{index}'].grid(row = 1+index, column = 1, sticky=tk.N)
        print(result)

        
    # 執行主程式
    window.mainloop()
            

def configImg(file):
    global reader, window, od_model, plate_list_img, plate_list_img_label, plate_list_number_label, label_img
    image = Image.open(file)
    predictions = od_model.predict_image(image)
    print(predictions)
    print(file)
    img = cv2.imread(file)
    
    plate_list = []
    h, w, c = img.shape
    for ret in predictions:
        prob = ret['probability']
        if prob < 0.5: continue
        tagID = ret['tagId']
        tagName = ret['tagName']
        bbox = ret['boundingBox']
        left = bbox['left']
        top = bbox['top']
        width = bbox['width']
        height = bbox['height']
        x1 = int(left*w)
        y1 = int(top*h)
        x2 = x1 + int(width*w)
        y2 = y1 + int(height*h)
        p0 = (max(x1, 15), max(y1-5, 15))
        print(f"probability is {prob}, tag id is {tagID}, tag name is {tagName}")
        print(f"bounding box ({x1}, {x2}, {y1}, {y2})")
        info = "{:.2f}:-{}".format(prob, tagName)
        cv2.rectangle(img, (x1, y1) , (x2, y2), (0, 0, 0), 10)
        cv2.rectangle(img, (x1, y1) , (x1+400, y1-50), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, info, p0, cv2.FONT_ITALIC, 2, (0, 255, 0), 3)
        img_copy = img.copy()
        plate = img_copy[y1:y2, x1:x2]
        plate_list.append(plate)
    
    #Rearrange colors
    blue,green,red = cv2.split(img)
    img = cv2.merge((red,green,blue))
    img = Image.fromarray(img)

    for index, img_plate in enumerate(plate_list):
        result = reader.readtext(img_plate, detail = 0)
        plate_list_number_label[f'n{index}'].configure(text=result)
        #Rearrange colors
        blue,green,red = cv2.split(img_plate)
        img_plate = cv2.merge((red,green,blue))
        img_plate = Image.fromarray(img_plate)
        plate_list_img[f'l{index}'] = ImageTk.PhotoImage(img_plate.resize((160, 90)))
        plate_list_img_label[f'i{index}'].configure(image=plate_list_img[f'l{index}'])
        img = ImageTk.PhotoImage(img.resize((640, 360)))
        label_img.configure(image=img)
        print(result)
    

def openImg(filename):
    global window, plate_list_img, plate_list_img_label, plate_list_number_label, label_img
    image = Image.open(filename)
    predictions = od_model.predict_image(image)
    print(predictions)
    print(filename)
    img = cv2.imread(filename)
    
    plate_list = []
    h, w, c = img.shape
    for ret in predictions:
        prob = ret['probability']
        if prob < 0.5: continue
        tagID = ret['tagId']
        tagName = ret['tagName']
        bbox = ret['boundingBox']
        left = bbox['left']
        top = bbox['top']
        width = bbox['width']
        height = bbox['height']
        x1 = int(left*w)
        y1 = int(top*h)
        x2 = x1 + int(width*w)
        y2 = y1 + int(height*h)
        p0 = (max(x1, 15), max(y1-5, 15))
        print(f"probability is {prob}, tag id is {tagID}, tag name is {tagName}")
        print(f"bounding box ({x1}, {x2}, {y1}, {y2})")
        info = "{:.2f}:-{}".format(prob, tagName)
        cv2.rectangle(img, (x1, y1) , (x2, y2), (0, 0, 0), 10)
        cv2.rectangle(img, (x1, y1) , (x1+400, y1-50), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, info, p0, cv2.FONT_ITALIC, 2, (0, 255, 0), 3)
        img_copy = img.copy()
        plate = img_copy[y1:y2, x1:x2]
        plate_list.append(plate)
    # cv2.imshow("Object-detection", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #Rearrange colors
    blue,green,red = cv2.split(img)
    img = cv2.merge((red,green,blue))
    img = Image.fromarray(img)
    
    img = ImageTk.PhotoImage(img.resize((640, 360)))
    label_img = tk.Label(window, image=img, bg = "blue")
    label_img.grid(row = 1, column = 0, rowspan = 5, sticky=tk.N)
    for index, img_plate in enumerate(plate_list):
        result = reader.readtext(img_plate, detail = 0)
        plate_list_number_label[f'n{index}'] = tk.Label(window, text=result, bg = "red")
        plate_list_number_label[f'n{index}'].grid(row = 1+index, column = 2, sticky=tk.N)
        #Rearrange colors
        blue,green,red = cv2.split(img_plate)
        img_plate = cv2.merge((red,green,blue))
        img_plate = Image.fromarray(img_plate)
        plate_list_img[f'l{index}'] = ImageTk.PhotoImage(img_plate.resize((160, 90)))
        plate_list_img_label[f'i{index}'] = tk.Label(window, image=plate_list_img[f'l{index}'], bg = "red")
        plate_list_img_label[f'i{index}'].grid(row = 1+index, column = 1, sticky=tk.N)
        print(result)
main()

