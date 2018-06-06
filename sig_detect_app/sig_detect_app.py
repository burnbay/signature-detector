import os, glob, re
import numpy as np
import random
from flask import Flask, flash, request, redirect, url_for
from flask import make_response
from flask import jsonify
from flask import Response
from werkzeug.utils import secure_filename
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from bs4 import BeautifulSoup as soup
import img2pdf
import csv

ALLOWED_EXTENSIONS = set(['pdf',"png","jpg","jpeg"]) # We only allow upload of pdf and image files.

app = Flask(__name__)


def increase_contrast(file_name,factor=20):
    img = Image.open(file_name).convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    rgbimg = Image.new("RGBA", img.size)
    rgbimg.paste(img)
    rgbimg.save(file_name)
    return


def hocr_img(img_file_name, temp_file_name = "temp_pdf_png"):
    os.system("rm {0}*".format(temp_file_name))
    
    file = img_file_name.rsplit('.', 1)[0]
    suffix = img_file_name.rsplit('.', 1)[1]

    os.system("convert -density 300 -depth 8 -quality 85 {0} {1}-0.png".format(img_file_name, temp_file_name))
    os.system("tesseract -l nor {0} {1}-0 hocr".format(img_file_name, temp_file_name))
    #increase_contrast(img_file_name) 

    hocr_files = ["{0}-0.hocr".format(temp_file_name)]
    return(hocr_files)


def hocr_pdf(pdf_file_name,temp_file_name = "temp_pdf_png"):
    os.system("rm {0}*".format(temp_file_name))

    os.system("convert -density 300 -depth 8 -quality 85 {0} {1}.png".format(pdf_file_name, temp_file_name))
    png_files = glob.glob("{0}*.png".format(temp_file_name))
    
    for png_file in png_files:
        file = png_file.replace(".png","")
        os.system("tesseract -l nor {0} {1} hocr".format(png_file, file))
        #increase_contrast(png_file)

    if os.path.isfile("{0}.png".format(temp_file_name)):
        os.system("mv {0}.png {0}-0.png".format(temp_file_name))
        os.system("mv {0}.hocr {0}-0.hocr".format(temp_file_name))
    
    hocr_files = sorted(glob.glob("{0}*.hocr".format(temp_file_name)))
    
    return(hocr_files)


def search_hocr_for_sentence(sentence, trigger_dict, hocr, page_num):
    
    word_span_list = hocr.find_all("span", {"id" : lambda L: L and L.startswith('word_')})
    
    word_list = []
    word_id_list = []
    bbox_list = []
    
    for word_span in word_span_list:
        #word = re.sub(r'[^\w\s]','',word_span.find(text=True))
        word = re.sub(";",":", word_span.find(text=True))
        word_id = int(word_span["id"].replace("word_1_",""))
        bbox = word_span["title"].replace(";","").split()[1:5]
        
        word_list.append(word)
        word_id_list.append(word_id)
        bbox_list.append(bbox)
    
    sentence_words = sentence.split()
    sentence_length=len(sentence_words)
    
    n=0
    for i in range(len(word_list)):
        if sentence_words[0] in word_list[i]:
            candidate_sentence = " ".join(word_list[i:(i+sentence_length)])
            if sentence in candidate_sentence:
                n+=1
                bbox = bbox_list[i+sentence_length-1]
                trigger_dict.update({sentence+" "+str(n) : {"page" : page_num,
                                                            "x0":int(bbox[0]),
                                                            "y0":int(bbox[1]),
                                                            "x1":int(bbox[2]),
                                                            "y1":int(bbox[3])}})
    return(trigger_dict, word_list, word_id_list, bbox_list)


def get_trigger_areas(hocr_list,
                      trigger_sentences = "trigger_sentences.csv"):

    with open(trigger_sentences) as csvfile:
        reader = csv.reader(csvfile)
        trigger_sentences = list(reader)[0]
    
    trigger_dict = {}
    
    page_num=0
    for hocr_file in hocr_list:
        page_num+=1
        with open(hocr_file,"r") as file:
            hocr = soup(file, "html5lib")
        
        for trigger_sentence in trigger_sentences:
            trigger_dict, word_list, word_id_list, bbox_list = search_hocr_for_sentence(trigger_sentence,
                                                                                        trigger_dict,
                                                                                        hocr,
                                                                                        page_num)
    return(trigger_dict, word_list, word_id_list, bbox_list)


def predict_image(model_name, img, threshold=0.5, show=False):
    """Use a pretrained CNN to judge if an image contains a signature."""

    img = img.convert('RGB')
    
    img = img.resize((150, 150), Image.ANTIALIAS)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()
    
    K.clear_session()
    model = load_model(model_name)
    
    pred = model.predict(img_tensor)
    
    if pred[0]<threshold:
        conclusion = False # "Not signed!"
        confidence = (threshold-pred)/threshold
    else:
        conclusion = True # "Signed!"
        confidence = (pred-threshold)/(1-threshold)
    
    return conclusion, np.round(confidence,4)


def score_signatures(trigger_dict, 
                     rel_left=3,
                     rel_right=10,
                     rel_up=0,
                     rel_down=6,
                     model="model.h5",
                     temp_file_name="temp_pdf_png"):
    """Use a pretrained CNN to search for signatures in each trigger area in trigger_dict. The results are added to trigger_dict and the latter returned."""
    
    for trigger in trigger_dict:
        page_file  = temp_file_name + "-" + str(trigger_dict[trigger]["page"]-1) + ".png"
        page_img   = Image.open(page_file)
        page_width = page_img.size[0]
        page_height= page_img.size[1]
        
        x0 = trigger_dict[trigger]["x0"]
        x1 = trigger_dict[trigger]["x1"]
        y0 = trigger_dict[trigger]["y0"]
        y1 = trigger_dict[trigger]["y1"]
        
        font_height = y1-y0
        
        x0 = np.max([x0 - font_height*rel_left, 0])
        x1 = np.min([x0 + font_height*rel_right, page_width])
        y0 = np.max([y0 - font_height*rel_up, 0])
        y1 = np.min([y0 + font_height*rel_down, page_height])
        
        crop_img = page_img.crop((x0,y0,x1,y1))
        
        signed, confidence = predict_image(model,crop_img,show=False)
        
        trigger_dict[trigger].update({"x0":int(x0),
                                      "x1":int(x1),
                                      "y0":int(y0),
                                      "y1":int(y1),
                                      "signed":signed, 
                                      "confidence":np.float(confidence)})
    
    return(trigger_dict)


def score(hocr_files):
    trigger_dict,_,_,_ = get_trigger_areas(hocr_files)
    score_dict = score_signatures(trigger_dict)
    
    return(score_dict)


def visualize_detections(trigger_dict, filename, fnt_size=30, temp_file_name = "temp_pdf_png"):
    
    png_file_dict = {}
    for trigger in trigger_dict:
        
        page_file  = temp_file_name + "-" + str(trigger_dict[trigger]["page"]-1) + ".png"
        img = Image.open(page_file).convert("RGB")
        drawing = ImageDraw.Draw(img)
        
        top_left = (trigger_dict[trigger]["x0"],trigger_dict[trigger]["y0"])
        bottom_right= (trigger_dict[trigger]["x1"],trigger_dict[trigger]["y1"])

        text_start_x = trigger_dict[trigger]["x0"]
        text_start_y = trigger_dict[trigger]["y1"]-fnt_size
        text_start = (text_start_x, text_start_y)
        
        txt_fnt  = ImageFont.truetype("DejaVuSerif.ttf", fnt_size)

        if trigger_dict[trigger]["signed"]==True:
            color="green"
            text = str(trigger)+" Signed! Conf.: " + str(np.round(trigger_dict[trigger]["confidence"]))
        else:
            color="red"
            text = str(trigger)+" Not signed! Conf.: " + str(np.round(trigger_dict[trigger]["confidence"],3))

        drawing.rectangle([top_left, bottom_right], outline=color)
        drawing.text(text_start, text, font=txt_fnt, fill=color)
        
        img.save(page_file)
        
    img_list = sorted(glob.glob("{0}*.png".format(temp_file_name)))
    #with open(filename, "wb") as f:
    #    f.write(img2pdf.convert(img_list))
    pdf_resp = img2pdf.convert(img_list)
    resp = Response(pdf_resp)
    resp.headers['Content-Disposition'] = "inline; filename=%s" % filename
    resp.mimetype = 'application/pdf'

    return (resp)


def remove_non_ascii(s): return "".join(i for i in s if ord(i)<128)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # if we got a valid file:
        
        if file and allowed_file(file.filename):
            
            filename = remove_non_ascii(secure_filename(file.filename))        
            
            file.save(filename)
            
            # if we got an image file:
            if filename.rsplit('.', 1)[1].lower() != "pdf":
                hocr_files = hocr_img(filename)
            
            # if we got a pdf file:
            if filename.rsplit('.', 1)[1].lower() == "pdf":
                hocr_files = hocr_pdf(filename)
            
            response_dict = score(hocr_files)
        
            if request.form.get('json'):
                return(jsonify(response_dict))
            
            else:
                response_pdf = visualize_detections(response_dict,filename)
                return(response_pdf)
        
    return '''
    <!doctype html>
    <title>Signature detection app</title>
    
    <h1>Signature detector v0.1</h2>
    Upload PDF/PNG/JPG-file to search for signatures:<br>
    
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload><br><br>
      Return results as JSON-file: <input type=checkbox name=json>
    </form>
    <br>
    <i>(Large multipage files may take a while, be patient.)</i>
    '''

if __name__ == "__main__":
    app.run()
